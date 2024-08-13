import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, VQVAE
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
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
from monai.apps import MedNISTDataset
import torch.nn as nn
from util_transformations import get_upsampler_dataloader
from monai.networks.nets import VarAutoEncoder

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
    parser.add_argument("--training_ids", help="Location of file with train ids.")

    args = parser.parse_args()

    set_determinism(seed=args.seed)
    #print_config()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating model...")
    device = torch.device("cuda")
    config = OmegaConf.load(args.stage1_config_file_path)
    stage1 = VQVAE(**config["stage1"]["params"])
    #stage1 = VarAutoEncoder(**config["stage1"]["params"])
    stage1.load_state_dict(torch.load(args.stage1_path))
    stage1 = stage1.to(device)
    stage1.eval()
   
    train_loader, eval_loader = get_upsampler_dataloader(
        batch_size=1,
        training_ids=args.training_ids,
        validation_ids=args.test_ids,
        num_workers=4,
    )
    
    psnr_metric = PSNRMetric(max_val=1.0)
    mae_metric = MAEMetric()
    mssim_metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=7)
    ssim_metric = SSIMMetric(spatial_dims=2, kernel_size=7)
    
    psnr_total = 0
    mae_total = 0
    mssim_total = 0
    ssim_total = 0
    quant_imgs = 0
    
    for batch in tqdm(eval_loader, ncols=110):
                
        with torch.no_grad():
            image = batch['image'].to(device)
            reconstruction, quantization_loss = stage1(images=image)

        psnr_value = psnr_metric(image, reconstruction)
        mae_value = mae_metric(image, reconstruction)
        mssim_value = mssim_metric(image, reconstruction)
        ssim_value = ssim_metric(image, reconstruction)
        
        print('\n')
        print('MSSIM: ', mssim_value[0,0].item() )
        quant_imgs += image.shape[0]
        
        for idx, image_batch in enumerate(batch['image']):
            psnr_total += psnr_value[idx,0].item()
            mae_total += mae_value[idx,0].item()
            mssim_total += mssim_value[idx,0].item()
            ssim_total += ssim_value[idx,0].item()
   
            img_name = batch['image_meta_dict']['filename_or_obj'][idx].split('/')[-1].split(".dcm")[0]

            path_img_recon_tiff = os.path.join(output_dir, f"{img_name}_recons.tif")
            path_img_tiff = os.path.join(output_dir, f"{img_name}.tif")
            #path_img_chain_tiff = os.path.join(output_dir, f"{img_name}_chain.tif")

            # print(path_img_chain_tiff)
            img_recons = reconstruction[idx, 0].detach().cpu().numpy()
            img = batch['image'][idx, 0].cpu().numpy()
            #img_chain = chain[idx, 0].cpu().numpy()
            
            tifffile.imwrite(path_img_tiff, img)  
            tifffile.imwrite(path_img_recon_tiff, img_recons)              
            #tifffile.imwrite(path_img_chain_tiff, img_chain)
        
        torch.cuda.empty_cache()
    