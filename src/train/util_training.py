from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.losses.adversarial_loss import PatchAdversarialLoss
from pynvml.smi import nvidia_smi
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")

def train_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    device: torch.device,
    adv_weight: float,
    perceptual_weight: float,
    kl_weight: float,
    adv_start: int,
    output_dir: str
) -> float:
    
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    raw_model = model.module if hasattr(model, "module") else model

    # val_loss = eval_aekl(
    #     model=model,
    #     discriminator=discriminator,
    #     perceptual_loss=perceptual_loss,
    #     loader=val_loader,
    #     device=device,
    #     step=len(train_loader) * start_epoch,
    #     kl_weight=kl_weight,
    #     adv_weight=adv_weight if start_epoch >= adv_start else 0.0,
    #     perceptual_weight=perceptual_weight,
    # )
    #print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    
    for epoch in range(start_epoch, n_epochs):
        train_epoch_aekl(
            model=model,
            discriminator=discriminator,
            perceptual_loss=perceptual_loss,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            kl_weight=kl_weight,
            adv_weight=adv_weight if epoch >= adv_start else 0.0,
            perceptual_weight=perceptual_weight,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_aekl(
                model=model,
                discriminator=discriminator,
                perceptual_loss=perceptual_loss,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                kl_weight=kl_weight,
                adv_weight=adv_weight if epoch >= adv_start else 0.0,
                perceptual_weight=perceptual_weight,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss

    print(f"Training finished!")
    print(f"Saving final model...")
    model_save_path = os.path.join(output_dir, "aekl")
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(raw_model.state_dict(), os.path.join(model_save_path, "final_model_aekl.pth"))

    return val_loss


def train_epoch_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    kl_weight: float,
    adv_weight: float,
    perceptual_weight: float,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
) -> None:
    model.train()
    discriminator.train()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)

    pbar = tqdm(enumerate(loader), total=len(loader))
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    
    for step, x in pbar:
        images = x["image"].to(device)

        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())

            kl_loss = 0.5 * torch.sum(
                z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
                dim=[1, 2, 3],
            )
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # DISCRIMINATOR
        if adv_weight > 0:
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                d_loss = adv_weight * discriminator_loss
                d_loss = d_loss.mean()

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            scaler_d.step(optimizer_d)
            scaler_d.update()
            
        else:
            discriminator_loss = torch.tensor([0.0]).to(device)

        epoch_loss += l1_loss.item()
        if adv_weight > 0:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()
            
        losses["d_loss"] = discriminator_loss
        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "l1_loss": f"{epoch_loss:.6f}",
                "p_loss": f"{losses['p_loss'].item():.6f}",
                "g_loss": f"{gen_epoch_loss:.6f}",
                "d_loss": f"{disc_epoch_loss:.6f}",
            },
        )

@torch.no_grad()
def eval_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    kl_weight: float,
    adv_weight: float,
    perceptual_weight: float,
) -> float:
    model.eval()
    discriminator.eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    total_losses = OrderedDict()
    for x in loader:
        images = x["image"].to(device)

        with autocast(enabled=True):
            # GENERATOR
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(
                z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
                dim=[1, 2, 3],
            )
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # DISCRIMINATOR
            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            else:
                discriminator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()
            d_loss = discriminator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
                d_loss=d_loss,
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    return total_losses["l1_loss"]

def train_upsampler_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    low_res_scheduler: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    device: torch.device,
    output_dir:str,
    scale_factor: float = 1.0
) -> float:
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    # val_loss = eval_upsampler_ldm(
    #     model=model,
    #     stage1=stage1,
    #     scheduler=scheduler,
    #     low_res_scheduler=low_res_scheduler,
    #     loader=val_loader,
    #     device=device,
    #     step=len(train_loader) * start_epoch,
    #     scale_factor=scale_factor,
    # )
    # print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_upsampler_ldm(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            low_res_scheduler=low_res_scheduler,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            scale_factor=scale_factor,
        )
        torch.cuda.empty_cache()

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_upsampler_ldm(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                low_res_scheduler=low_res_scheduler,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                scale_factor=scale_factor,
            )
            torch.cuda.empty_cache()

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
            
    print(f"Training finished!")
    print(f"Saving final model...")
    model_save_path = os.path.join(output_dir, "ldm")
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(raw_model.state_dict(), os.path.join(model_save_path, "final_model_ldm.pth"))

    return val_loss


def train_epoch_upsampler_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    low_res_scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    scale_factor: float = 1.0,
) -> None:
    model.train()
    epoch_loss = 0

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["image"].to(device)
        low_res_image = x["low_res_image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                latent = stage1.encode_stage_2_inputs(images) * scale_factor

            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
            low_res_timesteps = torch.randint(0, 350, (low_res_image.shape[0],), device=device).long()
        
            noise = torch.randn_like(latent).to(device)
            low_res_noise = torch.randn_like(low_res_image).to(device)
            noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
            noisy_low_res_image = low_res_scheduler.add_noise(
                original_samples=low_res_image,
                noise=low_res_noise,
                timesteps=low_res_timesteps,
            )

            latent_model_input = torch.cat([noisy_latent, noisy_low_res_image], dim=1)

            noise_pred = model(
                x=latent_model_input,
                timesteps=timesteps,
                class_labels=low_res_timesteps,
            )
                
            loss = F.mse_loss(noise_pred.float(), noise.float())

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{epoch_loss / (step + 1)}"
            }
        )

@torch.no_grad()
def eval_upsampler_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    low_res_scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    scale_factor: float = 1.0,
) -> float:
    
    model.eval()
    total_losses = OrderedDict()

    for x in loader:
        images = x["image"].to(device)
        low_res_image = x["low_res_image"].to(device)

        with autocast(enabled=True):
            latent = stage1.encode_stage_2_inputs(images) * scale_factor

            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
            low_res_timesteps = torch.randint(0, 350, (low_res_image.shape[0],), device=device).long()

            noise = torch.randn_like(latent).to(device)
            low_res_noise = torch.randn_like(low_res_image).to(device)
            noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
            noisy_low_res_image = low_res_scheduler.add_noise(
                original_samples=low_res_image,
                noise=low_res_noise,
                timesteps=low_res_timesteps,
            )
    
            latent_model_input = torch.cat([noisy_latent, noisy_low_res_image], dim=1)

            noise_pred = model(
                x=latent_model_input,
                timesteps=timesteps,
                class_labels=low_res_timesteps,
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    return total_losses["loss"]


