import argparse
import warnings
import os

import torch
import torch.nn as nn
import torch.optim as optim
from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.vqvae import VQVAE

from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism, first
from omegaconf import OmegaConf
import wandb
from torch.cuda.amp import autocast
import yaml
from util_training import train_upsampler_ldm
from util_transformations import get_upsampler_dataloader, get_upsampler_dataloader_mednist
import torchvision.models as models
from transformers import CLIPTextModel

warnings.filterwarnings("ignore")

class ResNetEncoder(nn.Module):
    def __init__(self, resnet):
        super(ResNetEncoder, self).__init__()
        self.resnet = resnet
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x
    
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
    parser.add_argument("--checkpoint_path", help="Number of loader workers")

    args = parser.parse_args()
        
    set_determinism(seed=args.seed)
    print_config()
    
    output_dir = "/project/outputs/runs_final/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Carregando os dados...")
    train_loader, val_loader = get_upsampler_dataloader(
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
    )
    
    # train_loader, val_loader = get_upsampler_dataloader_mednist(
    #     batch_size=args.batch_size,
    #     training_ids=args.training_ids,
    #     validation_ids=args.validation_ids,
    #     num_workers=args.num_workers,
    # )
    # Load Autoencoder to produce the latent representations
    print(f"Carregando o autoencoder(Stage 1) em {args.stage1_path}")
    
    config_autoencoder = OmegaConf.load(args.stage1_config_file_path)
    stage1 = AutoencoderKL(**config_autoencoder["stage1"]["params"])
    #stage1 = VQVAE(**config_autoencoder["stage1"]["params"])
    
    with open(args.config_file, 'r') as file:
        config_wandb = yaml.safe_load(file)
    
    #run = wandb.init(project="Artigo", name="LDM-Clinical-Zero-Padding", config=config_wandb)
    run = wandb.init(project="Artigo", name="LDM-Clinical-Lung-Multiclass-Mask", config=config_wandb)

    stage1.load_state_dict(torch.load(args.stage1_path))
    stage1.eval()

    # Create the diffusion model
    print("Carregando o modelo...")
    config = OmegaConf.load(args.config_file)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    if args.checkpoint_path:
        print('Carregando checkpoint...')
        diffusion.load_state_dict(torch.load(args.checkpoint_path))
        
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    low_res_scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    
    # for lung mask
    # resnet = models.resnet18(pretrained=True)
    # modules = list(resnet.children())[:-1]
    # resnet = nn.Sequential(*modules)
    # resnet_encoder = ResNetEncoder(resnet)

    # Text model
    #text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")


    print(f"{torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    # if torch.cuda.device_count() > 1:
    #     stage1 = torch.nn.DataParallel(stage1)
    #     diffusion = torch.nn.DataParallel(diffusion)

    stage1 = stage1.to(device)
    diffusion = diffusion.to(device)
    
    #text_encoder = text_encoder.to(device)

    # for lung mask
    #resnet_encoder = resnet_encoder.to(device)
    
    check_data = first(train_loader)

    with torch.no_grad():
        with autocast(enabled=True):
            z = stage1.encode_stage_2_inputs(check_data["image"].to('cuda'))
            #z = stage1.encode(check_data["image"].to('cuda'))
    
    print(f'média: {torch.mean(z)}')
    print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    
    optimizer = optim.Adam(diffusion.parameters(), lr=config["ldm"]["base_lr"])
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
        scale_factor=scale_factor, #args.scale_factor
        run=run,
        resnet_encoder=None,
        text_encoder=None
    )