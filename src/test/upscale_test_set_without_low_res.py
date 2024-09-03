import argparse
import os
import pandas as pd
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile
from generative.metrics import SSIMMetric, MultiScaleSSIMMetric
from monai.metrics import MAEMetric, PSNRMetric
from torch.cuda.amp import autocast
from generative.inferers import LatentDiffusionInferer
from monai.apps import MedNISTDataset
from util_transformations import get_upsampler_dataloader
from monai.utils import set_determinism, first

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location to save the output.")
    parser.add_argument("--stage1_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--stage1_config_file_path", help="Path to the config file from the stage1.")
    parser.add_argument("--diffusion_config_file_path", help="Path to the config file from the diffusion model.")
    parser.add_argument("--start_index", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--stop_index", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--x_size", type=int, default=64, help="Latent space x size.")
    parser.add_argument("--y_size", type=int, default=64, help="Latent space y size.")
    parser.add_argument("--scale_factor", type=float, help="Latent space y size.")
    parser.add_argument("--num_inference_steps", type=int, help="")
    parser.add_argument("--noise_level", type=int, help="")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--training_ids", help="Location of file with test ids.")

    args = parser.parse_args()

    #set_determinism(seed=args.seed)
    print_config()

    output_dir = args.output_dir
    #output_dir = '/project/src/train'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating model...")
    device = torch.device("cuda")
    config = OmegaConf.load(args.stage1_config_file_path)
    stage1 = AutoencoderKL(**config["stage1"]["params"])
    stage1.load_state_dict(torch.load(args.stage1_path))
    stage1 = stage1.to(device)
    stage1.eval()

    config = OmegaConf.load(args.diffusion_config_file_path)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    diffusion.load_state_dict(torch.load(args.diffusion_path))
    diffusion = diffusion.to(device)
    diffusion.eval()

    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    scheduler.set_timesteps(args.num_inference_steps)
    
    #df = pd.read_csv(args.test_ids, sep="\t")
    #df = df[args.start_index : args.stop_index]
    
    # samples_dir = os.listdir(args.downsampled_dir)
    # samples_datalist = []
    
    # for sample_path in sorted(samples_dir):
    #     samples_datalist.append(
    #         {
    #             "low_res_image": str(sample_path),
    #         }
    #     )
        
    # data_dicts = []
    # for index, row in df.iterrows():
    #     if '.tif' in row['image'] or '.tif' in row['image']:
    #         print(row['image'])
    #         continue
        
    #     data_dicts.append(
    #         {
    #             "image": str(row["image"]),
    #             "low_res_image": str(row['low_res_image'])
    #         }
    #     )
    # print(f"{len(data_dicts)} imagens.")
    
    # roi_image_size = 416
    
    # eval_transforms = transforms.Compose(
    #     [
    #       transforms.LoadImaged(keys=["image"]),
    #       transforms.EnsureChannelFirstd(keys=["image"]),
    #       transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    #       transforms.CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size,roi_image_size)),
    #       transforms.ToTensord(keys=["image"]),
    #     ]
    # )
    train_loader, eval_loader = get_upsampler_dataloader(
        batch_size=1,
        training_ids=args.training_ids,
        validation_ids=args.test_ids,
        num_workers=4,
    )
    check_data = first(train_loader)

    with torch.no_grad():
        with autocast(enabled=True):
            z = stage1.encode_stage_2_inputs(check_data["image"].to('cuda'))
            #z = stage1.encode(check_data["image"].to('cuda'))

    print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    #scale_factor = args.scale_factor
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    psnr_metric = PSNRMetric(max_val=1.0)
    mae_metric = MAEMetric()
    mssim_metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=4)
    ssim_metric = SSIMMetric(spatial_dims=2, kernel_size=4)
    
    psnr_total = 0
    mae_total = 0
    mssim_total = 0
    ssim_total = 0
    quant_imgs = 0
    
    for batch in tqdm(eval_loader, ncols=110):
        image = batch['image'].to(device)
        mask_path = batch['filename'][0] + '_mask.tiff'
        mask_img = tifffile.imread(os.path.join('/project/data_lung_multiclass_masks', mask_path))
        mask_tensor = torch.tensor(mask_img, dtype=torch.float32).unsqueeze(0).to(image.device)
        
        #latents = torch.randn((image.shape[0], config["ldm"]["params"]["out_channels"], args.x_size, args.y_size)).to(
        #    device
        #)
        latent = stage1.encode_stage_2_inputs(image) * inferer.scale_factor
        noise = torch.randn_like(latent).to(device)
        timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (latent.shape[0],), device=latent.device
            ).long()
        noisy_latent = inferer.scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)

        with torch.no_grad():
            decoded = inferer.sample(
                input_noise=noisy_latent,
                diffusion_model=diffusion,
                scheduler=scheduler,
                save_intermediates=False,
                intermediate_steps=100,
                autoencoder_model=stage1,
                conditioning=mask_tensor
            )
                
        psnr_value = psnr_metric(image, decoded)
        mae_value = mae_metric(image, decoded)
        mssim_value = mssim_metric(image, decoded)
        ssim_value = ssim_metric(image, decoded)
        
        print('MSSIM: ', mssim_value[0,0].item() )
        print('PSNR: ', psnr_value[0,0].item() )
        quant_imgs += image.shape[0]

        for idx, image_batch in enumerate(batch['image']):
            psnr_total += psnr_value[idx,0].item()
            mae_total += mae_value[idx,0].item()
            mssim_total += mssim_value[idx,0].item()
            ssim_total += ssim_value[idx,0].item()

            #img_name = batch['image_meta_dict']['filename_or_obj'][idx].split('/')[-1].split(".dcm")[0]
            path_img_up_tiff = os.path.join(output_dir, f"{ batch['filename'][0]}_upscale.tif")
            path_img_tiff = os.path.join(output_dir, f"{ batch['filename'][0]}.tif")
            print(path_img_tiff)
            img_upscale = decoded[idx, 0].cpu().numpy()
            img = batch['image'][idx, 0].cpu().numpy()
            
            tifffile.imwrite(path_img_tiff, img)  
            tifffile.imwrite(path_img_up_tiff, img_upscale)  
            
    output_logs = os.path.join(output_dir, 'logs')
    os.makedirs(output_logs, exist_ok=True)
    output_file_path = os.path.join(output_logs, 'log.txt')
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(f'Modelo: {args.diffusion_path}\n')
        file.write('Validação:\n')
        file.write('-------------------------------------------------\n')
        file.write(f"PSNR médio: {psnr_total / (quant_imgs + 1)}\n")
        file.write(f"MAE médio: {mae_total / (quant_imgs + 1)}\n")
        file.write(f"MSSIM médio: {mssim_total / (quant_imgs + 1)}\n")
        file.write(f"SSIM médio: {ssim_total / (quant_imgs + 1)}\n")
    
    print('PSNR: ',psnr_total / (quant_imgs + 1))
    print('MAE: ',mae_total / (quant_imgs + 1))
    print('MSSIM: ',mssim_total / (quant_imgs + 1))
    print('SSIM: ',ssim_total / (quant_imgs + 1))