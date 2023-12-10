""" Script to upscale the samples of the downsampled LDM."""
import argparse
import os
from pathlib import Path
import nibabel as nib
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
from monai.utils import set_determinism
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    parser.add_argument("--downsampled_dir", help='Location of img files.')
    
    args = parser.parse_args()

    set_determinism(seed=args.seed)
    print_config()
    
    output_dir = args.output_dir
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

    scheduler = DDIMScheduler(
        num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config["ldm"]["scheduler"]["beta_start"],
        beta_end=config["ldm"]["scheduler"]["beta_end"],
        schedule=config["ldm"]["scheduler"]["schedule"],
        prediction_type=config["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    scheduler.set_timesteps(args.num_inference_steps)

    # Samples
    samples_dir = os.listdir(args.downsampled_dir)
    samples_datalist = []
    
    for sample_path in sorted(samples_dir):
        samples_datalist.append(
            {
                "low_res_image": str(sample_path),
            }
        )
        
    print(f"{len(samples_datalist)} images found in {str(samples_dir)}")
    samples_datalist = samples_datalist[args.start_index : args.stop_index]

    low_res_size = 64
    sample_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["low_res_image"]),
            transforms.EnsureChannelFirstd(keys=["low_res_image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.Resized(keys=["low_res_image"], spatial_size=(low_res_size, low_res_size)),
            transforms.ToTensord(keys=["low_res_image"]),
        ]
    )

    samples_ds = Dataset(
        data=samples_datalist,
        transform=sample_transforms,
    )
    samples_loader = DataLoader(
        samples_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    for batch in tqdm(samples_loader):
        low_res_image = batch["low_res_image"].to(device)

        latents = torch.randn((1, config["ldm"]["params"]["out_channels"], args.x_size, args.y_size)).to(
            device
        )
        low_res_noise = torch.randn((1, 1, args.x_size, args.y_size)).to(device)

        noise_level = torch.Tensor((args.noise_level,)).long().to(device)
        noisy_low_res_image = scheduler.add_noise(
            original_samples=low_res_image,
            noise=low_res_noise,
            timesteps=torch.Tensor((noise_level,)).long().to(device),
        )
        
        scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)
        for t in tqdm(scheduler.timesteps, ncols=110):
            with torch.no_grad():
                latent_model_input = torch.cat([latents, noisy_low_res_image], dim=1)
                noise_pred = diffusion(
                    x=latent_model_input,
                    timesteps=torch.Tensor((t,)).to(device),
                    class_labels=noise_level,
                )
                latents, _ = scheduler.step(noise_pred, t, latents)

        with torch.no_grad():
            sample = stage1.decode_stage_2_outputs(latents / args.scale_factor)

        img_name = batch["image_meta_dict"]["filename_or_obj"][0]
        
        path_img = os.path.join(output_dir, img_name)
        path_img_down = os.path.join(output_dir, img_name + '_downscale')
        path_img_up = os.path.join(output_dir, img_name + '_upscale')
        
        mpimg.imsave(path_img + '.png', batch['image'][0,0], cmap=plt.cm.gray)
        mpimg.imsave(path_img_down + '.png', low_res_image[0,0], cmap=plt.cm.gray)
        mpimg.imsave(path_img_up + '.png', sample[0,0], cmap=plt.cm.gray)