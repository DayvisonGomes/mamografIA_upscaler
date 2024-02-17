import warnings
import torch
import argparse
import os
#import wandb
import torch.optim as optim

from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from util_training import train_aekl
from util_transformations import get_upsampler_dataloader

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Caminho do arquivo de configuração .yaml.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--autoencoder_warm_up_n_epochs",type=int,default=10,help="Epoch when the adversarial training starts.",)
    parser.add_argument("--val_interval",type=int,default=10,help="Number of epochs to between evaluations.",)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    
    #run = wandb.init(project="Tcc", name="AutoencoderKL")

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
    
    print("Criando o modelo...")
    config = OmegaConf.load(args.config_file)
    model = AutoencoderKL(**config["stage1"]["params"])
    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"])

    print(f"{torch.cuda.device_count()} GPUs")
    device = torch.device("cuda")
    model = model.to(device)
    perceptual_loss = perceptual_loss.to(device)
    discriminator = discriminator.to(device)
    
    optimizer_g = optim.Adam(model.parameters(), lr=config["stage1"]["base_lr"])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config["stage1"]["disc_lr"])

    best_loss = float("inf")
    start_epoch = 0
    
    print(f"Començando o treinamento")
    val_loss = train_aekl(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        n_epochs=args.n_epochs,
        eval_freq=args.val_interval,
        device=device,
        kl_weight=config["stage1"]["kl_weight"],
        adv_weight=config["stage1"]["adv_weight"],
        perceptual_weight=config["stage1"]["perceptual_weight"],
        adv_start=args.autoencoder_warm_up_n_epochs,
        output_dir=output_dir,
        #run=run
    )