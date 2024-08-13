import warnings
import torch
import argparse
import os
#import wandb
import torch.optim as optim

from generative.losses.perceptual import PerceptualLoss
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from util_training_vqvae import train_vqvae
from util_transformations import get_upsampler_dataloader, get_upsampler_dataloader_mednist, get_upsampler_dataloader_mednist_2dldm
from generative.networks.nets import VQVAE
from torch.nn import L1Loss

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Caminho do arquivo de configuração .yaml.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--val_interval",type=int,default=10,help="Number of epochs to between evaluations.",)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--checkpoint_path", help="Checkpoint path")

    #run = wandb.init(project="Artigo", name="AutoencoderKL")

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
    # train_loader, val_loader = get_upsampler_dataloader_mednist_2dldm(
    #     batch_size=args.batch_size,
    #     training_ids=args.training_ids,
    #     validation_ids=args.validation_ids,
    #     num_workers=args.num_workers,
    # )
    
    print("Criando o modelo...")
    config = OmegaConf.load(args.config_file)
    model = VQVAE(**config["stage1"]["params"]) # uma forma de upar o arquivo yaml pro wandb
    
    if args.checkpoint_path:
        print('Carregando checkpoint...')
        model.load_state_dict(torch.load(args.checkpoint_path))
    
    l1_loss = L1Loss()
    discriminator = PatchDiscriminator(spatial_dims=2, in_channels=1, num_layers_d=3, num_channels=64)
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    perceptual_weight = 0.001

    print(f"{torch.cuda.device_count()} GPUs")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        l1_loss = torch.nn.DataParallel(l1_loss)
    
    model = model.to(device)
    l1_loss = l1_loss.to(device)
    perceptual_loss.to(device)
    discriminator.to(device)

    optimizer_g = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=5e-4)
            
    print(f"Començando o treinamento")
    val_loss = train_vqvae(
        model=model,
        l1_loss=l1_loss,
        perceptual_loss=perceptual_loss,
        discriminator=discriminator,
        adv_loss=adv_loss,
        adv_weight=adv_weight,
        perceptual_weight=perceptual_weight,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        n_epochs=args.n_epochs,
        eval_freq=args.val_interval,
        device=device,
    )
    
    model_save_path = os.path.join(output_dir, "vqvae")
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, "vqvae_gan_v2.pth"))
