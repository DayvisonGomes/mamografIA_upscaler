import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism, first
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import tifffile
from generative.metrics import SSIMMetric, MultiScaleSSIMMetric
from monai.metrics import MAEMetric, PSNRMetric
from torch.cuda.amp import autocast

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

    args = parser.parse_args()

    set_determinism(seed=args.seed)
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

    scheduler = DDIMScheduler(**config["ldm"].get("scheduler", dict()))
    scheduler.set_timesteps(args.num_inference_steps)

    df = pd.read_csv(args.test_ids, sep="\t")
    #df = df[args.start_index : args.stop_index]

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": str(row["image"]),
                #"low_res_image": str(row['low_res_image'])
            }
        )
    print(f"{len(data_dicts)} imagens.")
    
    roi_image_size = 416
    roi_low_res_size = 291
    low_res_size = 208
    
    eval_transforms = transforms.Compose(
        [
        #   transforms.LoadImaged(keys=["image", "low_res_image"]),
        #   transforms.EnsureChannelFirstd(keys=["image", "low_res_image"]),
        #   transforms.ScaleIntensityd(keys=["image", "low_res_image"], minv=0.0, maxv=1.0),
          transforms.LoadImaged(keys=["image"]),
          transforms.EnsureChannelFirstd(keys=["image"]),
          transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
          transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),

          transforms.Resized(keys=["image"], spatial_size=(256, 256)),
          #transforms.CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size,roi_image_size)),
          #transforms.CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size,roi_low_res_size)),
          transforms.Resized(keys=["low_res_image"], spatial_size=(128, 128)),
          transforms.ToTensord(keys=["image", "low_res_image"]),
        ]
    )

    eval_ds = Dataset(
        data=data_dicts,
        transform=eval_transforms,
    )
    batch_size = 1
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    #with torch.no_grad():
    #    with autocast(enabled=True):
    #        z = stage1.encode_stage_2_inputs(first(eval_loader)["image"].to('cuda'))

    #print(f"Scaling factor set to {1/torch.std(z)}")
    #scale_factor = 1 / torch.std(z)
    scale_factor = args.scale_factor
    
    psnr_metric = PSNRMetric(max_val=1.0)
    mae_metric = MAEMetric()
    mssim_metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=7)
    ssim_metric = SSIMMetric(spatial_dims=2, kernel_size=7)
    
    psnr_total = 0
    mae_total = 0
    mssim_total = 0
    ssim_total = 0
    quant_imgs = 0
    
    for batch in tqdm(eval_loader):
        low_res_image = batch["low_res_image"].to(device)
        image = batch['image'].to(device)
        
        latents = torch.randn((low_res_image.shape[0], config["ldm"]["params"]["out_channels"], args.x_size, args.y_size)).to(
            device
        )
        low_res_noise = torch.randn((low_res_image.shape[0], 1, args.x_size, args.y_size)).to(device)

        noise_level = torch.Tensor((args.noise_level,)).long().to(device)
        noisy_low_res_image = scheduler.add_noise(
            original_samples=low_res_image,
            noise=low_res_noise,
            timesteps=torch.Tensor((noise_level,)).long().to(device),
        )
        #noisy_low_res_image = F.pad(noisy_low_res_image, (0, 1, 0, 1))  

        scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)
        for t in tqdm(scheduler.timesteps, ncols=110):
            with torch.no_grad():
                #with autocast(enabled=True):
                latent_model_input = torch.cat([latents, noisy_low_res_image], dim=1)
                noise_pred = diffusion(
                    x=latent_model_input,
                    timesteps=torch.Tensor((t,)).to(device),
                    class_labels=noise_level,
                )
                latents, _ = scheduler.step(noise_pred, t, latents)

        with torch.no_grad():
            sample = stage1.decode_stage_2_outputs(latents / scale_factor)
        
        # plt.figure(figsize=(8,8))
        # plt.style.use("default")
        # plt.imshow(
        #     torch.cat([image[0, 0].cpu(), sample[0, 0].cpu()], dim=1),
        #     vmin=0,
        #     vmax=1,
        #     cmap="gray",
        # )
        # plt.tight_layout()
        # plt.axis("off")
        # plt.show()
        
        psnr_value = psnr_metric(image, sample)
        mae_value = mae_metric(image, sample)
        mssim_value = mssim_metric(image, sample)
        ssim_value = ssim_metric(image, sample)
        
        quant_imgs += low_res_image.shape[0]

        for idx, image_batch in enumerate(batch['image']):
            psnr_total += psnr_value[idx,0].item()
            mae_total += mae_value[idx,0].item()
            mssim_total += mssim_value[idx,0].item()
            ssim_total += ssim_value[idx,0].item()
            
            img_name = batch['image_meta_dict']['filename_or_obj'][idx].split('/')[-1].split(".dcm")[0]
            path_img_up_tiff = os.path.join(output_dir, f"{img_name}_upscale.tif")
            path_img_tiff = os.path.join(output_dir, f"{img_name}.tif")
            print(path_img_tiff)
            img_upscale = sample[idx, 0].cpu().numpy()
            img = batch['image'][idx, 0].cpu().numpy()
            
            tifffile.imwrite(path_img_tiff, img)  
            tifffile.imwrite(path_img_up_tiff, img_upscale)  
            
        #torch.cuda.empty_cache()
    
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