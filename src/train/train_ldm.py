import argparse
import warnings
import os

import torch
import torch.nn as nn
import torch.optim as optim
from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets import AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf

from util_training import train_upsampler_ldm
from util_transformations import get_upsampler_dataloader

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Caminho do arquivo .yaml com as configurações do modelo de difusão.")
    parser.add_argument("--stage1_config_file_path", help="Location of file with validation ids.")
    parser.add_argument("--stage1_path", help="Path readable by load_model.")
    parser.add_argument("--scale_factor", type=float, help="")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--val_interval",type=int,default=10,help="Number of epochs to between evaluations.",)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
        
    set_determinism(seed=args.seed)
    print_config()
    
    output_dir = "/project/outputs/runs/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Carregando os dados...")
    train_loader, val_loader = get_upsampler_dataloader(
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
    )
    # Load Autoencoder to produce the latent representations
    print(f"Carregando o autoencoder(Stage 1) em {args.stage1_path}")
    config_autoencoder = OmegaConf.load(args.stage1_config_file_path)
    stage1 = AutoencoderKL(**config_autoencoder["stage1"]["params"])
    stage1.load_state_dict(torch.load(args.stage1_path))
    stage1.eval()

    # Create the diffusion model
    print("Carregando o modelo...")
    config = OmegaConf.load(args.config_file)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    low_res_scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    
    print(f"{torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    # if torch.cuda.device_count() > 1:
    #     stage1 = torch.nn.DataParallel(stage1)
    #     diffusion = torch.nn.DataParallel(diffusion)

    stage1 = stage1.to(device)
    diffusion = diffusion.to(device)
    
    optimizer = optim.AdamW(diffusion.parameters(), lr=config["ldm"]["base_lr"])
    best_loss = float("inf")
    start_epoch = 0
    
    print(f"Começando treinamento...")
    val_loss = train_upsampler_ldm(
        model=diffusion,
        stage1=stage1,
        scheduler=scheduler,
        low_res_scheduler=low_res_scheduler,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        eval_freq=args.val_interval,
        device=device,
        output_dir=output_dir,
        scale_factor=args.scale_factor,
    )


