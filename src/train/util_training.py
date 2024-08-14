from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.losses.adversarial_loss import PatchAdversarialLoss
from pynvml.smi import nvidia_smi
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import wandb
import tifffile
from generative.metrics import MultiScaleSSIMMetric

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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
    output_dir: str,
    run
) -> float:
    
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    raw_model = model.module if hasattr(model, "module") else model
    model_save_path = os.path.join(output_dir, "aekl")
    os.makedirs(model_save_path, exist_ok=True)
    
    #early_stop = EarlyStopping(patience=7, verbose=True, path=os.path.join(model_save_path, "model_aekl_best_highLR_moreEpochs.pth"))
    early_stop = EarlyStopping(patience=7, verbose=True, path=os.path.join(model_save_path, "model_aekl_best_4_latent_pulmao_new_input.pth"))

    val_loss = eval_aekl(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        kl_weight=kl_weight,
        adv_weight=adv_weight if start_epoch >= adv_start else 0.0,
        perceptual_weight=perceptual_weight,
        run=run
    )
    early_stop(val_loss, model)
    print_gpu_memory_report()

    run.log({'recons_loss_val': val_loss})

    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    #torch.cuda.empty_cache()
    
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
            run=run
        )
        #torch.cuda.empty_cache()
        
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
                run=run
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()
            early_stop(val_loss, model)
            run.log({'recons_loss_val': val_loss})

            #torch.cuda.empty_cache()
            
            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                
            if early_stop.early_stop:
                print("Early stopping")
                break
            
    print(f"Training finished!")
    print(f"Saving final model...")
    
    #torch.save(raw_model.state_dict(), os.path.join(model_save_path, "model_aekl_highLR_moreEpochs.pth"))
    torch.save(raw_model.state_dict(), os.path.join(model_save_path, "model_aekl_4_latent_pulmao_new_input.pth"))

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
    run
) -> None:
    model.train()
    discriminator.train()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)

    pbar = tqdm(enumerate(loader), total=len(loader), ncols=110)
    pbar.set_description(f"Epoch {epoch}")

    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    quant_batch = 0
    
    for step, x in pbar:
        images = x["image"].to(device)
        quant_batch += images.shape[0]
        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = model(images)

            recons_loss = F.l1_loss(reconstruction.float(), images.float())
            #recons_loss = F.smooth_l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        if adv_weight > 0:
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

        epoch_loss += recons_loss.item()
        if adv_weight > 0:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        pbar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
        
    run.log({'recons_loss_train': epoch_loss / (quant_batch + 1)})
    run.log({'gen_loss': gen_epoch_loss / (quant_batch + 1)})
    run.log({'disc_loss': disc_epoch_loss / (quant_batch + 1)})
  
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
    run,
) -> float:
    model.eval()
    discriminator.eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    total_losses = OrderedDict()
    
    original_images = []
    reconstructed_images = []
    
    for x in loader:
        images = x["image"].to(device)
        
        with torch.no_grad():
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
            
            original_images.append(images.cpu())
            reconstructed_images.append(reconstruction.cpu())
            
    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    batch_idx = torch.randint(len(original_images), (1,)).item()
    image_idx = torch.randint(original_images[batch_idx].shape[0], (1,)).item()
    original_image = original_images[batch_idx][image_idx]
    reconstructed_image = reconstructed_images[batch_idx][image_idx]

    run.log({
        f"Original Image Epoch {step}": wandb.Image(original_image),
        f"Reconstructed Image Epoch {step}": wandb.Image(reconstructed_image),
        "Step": step
    })
    
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
    scale_factor: float,
    run,
    resnet_encoder,
    text_encoder
) -> float:
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model
    
    model_save_path = os.path.join(output_dir, "ldm")
    os.makedirs(model_save_path, exist_ok=True)
    #early_stop = EarlyStopping(patience=17, verbose=True, path=os.path.join(model_save_path, "final_model_ldm_best_final.pth"))
    early_stop = EarlyStopping(patience=21, verbose=True, path=os.path.join(model_save_path, "model_ldm_best_4_latent_pulmao_new_input_full_mask.pth"))
    #early_stop = EarlyStopping(patience=14, verbose=True, path=os.path.join(model_save_path, "model_ldm_best_0403.pth"))

    val_loss = eval_upsampler_ldm(
        model=model,
        stage1=stage1,
        scheduler=scheduler,
        low_res_scheduler=low_res_scheduler,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        scale_factor=scale_factor,
        run=run,
        resnet_encoder=resnet_encoder,
        text_encoder=text_encoder
    )
    early_stop(val_loss, model)
    run.log({'mse_val_loss': val_loss})

    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    #torch.cuda.empty_cache()        
    
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
            run=run,
            resnet_encoder=resnet_encoder,
            text_encoder=text_encoder
        )
        #torch.cuda.empty_cache()

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
                run=run,
                resnet_encoder=resnet_encoder,
                text_encoder=text_encoder
            )
            #torch.cuda.empty_cache()
            early_stop(val_loss, model)
            run.log({'mse_val_loss': val_loss})

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
            
            if early_stop.early_stop:
                print("Early stopping")
                break
            
    print(f"Training finished!")
    print(f"Saving final model...")
    
    #torch.save(raw_model.state_dict(), os.path.join(model_save_path, "final_model_ldm_final.pth"))
    #torch.save(raw_model.state_dict(), os.path.join(model_save_path, "model_ldm_0403.pth"))
    torch.save(raw_model.state_dict(), os.path.join(model_save_path, "model_ldm_4_latent_pulmao_new_input_full_mask.pth"))

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
    scale_factor: float,
    run,
    resnet_encoder,
    text_encoder
) -> None:
    model.train()
    stage1.eval()
    epoch_loss = 0
    quant_batch = 0
    
    pbar = tqdm(enumerate(loader), total=len(loader), ncols=110)
    
    for step, x in pbar:
        images = x["image"].to(device)
        #low_res_image = x["low_res_image"].to(device)
        quant_batch += images.shape[0]
        #reports = x["report"].to(device)
        full_mask = x['mask'].to(device)
        
        # for lung mask
        #mask_path = x['filename'][0] + '_mask.tiff'
        #mask_img = tifffile.imread(os.path.join('/project/data_lung_mask',mask_path))
        #closed_lung_mask_tensor = torch.tensor(mask_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        #closed_lung_mask_tensor = closed_lung_mask_tensor.repeat(1, 3, 1, 1).to(images.device)
        #
        
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()
        #low_res_timesteps = torch.randint(
          #      0, 350, (low_res_image.shape[0],), device=low_res_image.device
        #).long()
            
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                latent = stage1.encode_stage_2_inputs(images) * scale_factor
                #latent = stage1.encode(images) * scale_factor

            #prompt_embeds = text_encoder(reports.squeeze(1))
            #prompt_embeds = prompt_embeds[0]
            
            noise = torch.randn_like(latent).to(device)
           # low_res_noise = torch.randn_like(low_res_image).to(device)

            noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
           # noisy_low_res_image = low_res_scheduler.add_noise(
            #    original_samples=low_res_image, noise=low_res_noise, timesteps=low_res_timesteps
            #)

            #latent_model_input = torch.cat([noisy_latent, noisy_low_res_image], dim=1)
            latent_model_input = noisy_latent

            #latent_model_input = torch.nn.functional.pad(latent_model_input, (1, 1, 1, 1), mode='constant', value=0)
            #noise = torch.nn.functional.pad(noise, (1, 1, 1, 1), mode='constant', value=0)

            # for lung mask
            #output_resnet = resnet_encoder(closed_lung_mask_tensor)
            #output_resnet = output_resnet.view(-1, 512).unsqueeze(0)
            
            #noise_pred = model(x=latent_model_input, timesteps=timesteps, class_labels=low_res_timesteps, context=full_mask)
            noise_pred = model(x=latent_model_input, timesteps=timesteps, context=full_mask)
            
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{(epoch_loss / (step + 1)):.4f}"
            }
        )
    
    run.log({'mse_train_loss': epoch_loss / (quant_batch + 1)})
    
def eval_upsampler_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    low_res_scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step,
    scale_factor,
    run,
    resnet_encoder,
    text_encoder
) -> float:
    
    model.eval()
    stage1.eval()
    total_losses = OrderedDict()

    for x in loader:
        images = x["image"].to(device)
        #low_res_image = x["low_res_image"].to(device)
        #reports = x["report"].to(device)
        full_mask = x['mask'].to(device)
        # for lung mask
        #mask_path = x['filename'][0] + '_mask.tiff'
        #mask_img = tifffile.imread(os.path.join('/project/data_lung_mask',mask_path))
        #closed_lung_mask_tensor = torch.tensor(mask_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        #closed_lung_mask_tensor = closed_lung_mask_tensor.repeat(1, 3, 1, 1).to(images.device)
        #
        
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()
        #low_res_timesteps = torch.randint(
        #        0, 350, (low_res_image.shape[0],), device=low_res_image.device
        #).long()
        
        with torch.no_grad():
            with autocast(enabled=True):
                latent = stage1.encode_stage_2_inputs(images) * scale_factor
                #latent = stage1.encode(images) * scale_factor

                #prompt_embeds = text_encoder(reports.squeeze(1))
                #prompt_embeds = prompt_embeds[0]
            
                noise = torch.randn_like(latent).to(device)
                #low_res_noise = torch.randn_like(low_res_image).to(device)
                
                noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
                #noisy_low_res_image = low_res_scheduler.add_noise(
                #    original_samples=low_res_image, noise=low_res_noise, timesteps=low_res_timesteps
                #)
                
                #latent_model_input = torch.cat([noisy_latent, noisy_low_res_image], dim=1)
                latent_model_input = noisy_latent
                #latent_model_input = torch.nn.functional.pad(latent_model_input, (1, 1, 1, 1), mode='constant', value=0)
                #noise = torch.nn.functional.pad(noise, (1, 1, 1, 1), mode='constant', value=0)
                
                # for lung mask
                #output_resnet = resnet_encoder(closed_lung_mask_tensor)
                #output_resnet = output_resnet.view(-1, 512).unsqueeze(0)
                
                #noise_pred = model(x=latent_model_input, timesteps=timesteps, class_labels=low_res_timesteps,context=output_resnet)
                noise_pred = model(x=latent_model_input, timesteps=timesteps,context=full_mask)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            losses = OrderedDict(loss=loss)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]
    
    # Sampling image during training
    #sampling_image = low_res_image[0].unsqueeze(0)
    latents = torch.randn((1, 4, 512, 512)).to(device)
    #low_res_noise = torch.randn((1, 1, 512, 512)).to(device)
    noise_level = 1
    noise_level = torch.Tensor((noise_level,)).long().to(device)
    #noisy_low_res_image = scheduler.add_noise(
    #    original_samples=sampling_image,
    #    noise=low_res_noise,
    #    timesteps=torch.Tensor((noise_level,)).long().to(device),
    #)

    scheduler.set_timesteps(num_inference_steps=1000)
    for t in tqdm(scheduler.timesteps, ncols=110):
        with torch.no_grad():
            with autocast(enabled=True):
                #latent_model_input = torch.cat([latents, noisy_low_res_image], dim=1)
                latent_model_input = noisy_latent

                # noise_pred = model(
                #     x=latent_model_input, timesteps=torch.Tensor((t,)).to(device), class_labels=noise_level,
                #     context=full_mask
                # )
                noise_pred = model(
                    x=latent_model_input, timesteps=torch.Tensor((t,)).to(device),
                    context=full_mask
                )
                
            latents, _ = scheduler.step(noise_pred, t, latents)

    with torch.no_grad():
        decoded = stage1.decode_stage_2_outputs(latents / scale_factor)
    
    mssim_metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=7)
    mssim_value = mssim_metric(images, decoded)

    print('MSSIM:', mssim_value[0,0].item())
    run.log({
        f"Original Image Epoch {step}": wandb.Image(images[0, 0].cpu()),
        f"Upscale Image Epoch {step}": wandb.Image(decoded[0, 0].cpu().numpy().astype(np.float32)),
        "Step": step
    })
    
    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    return total_losses["loss"]


def train_upsampler_ldm_without_low_res(
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
    inferer
    #run    
) -> float:
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model
    
    model_save_path = os.path.join(output_dir, "ldm")
    os.makedirs(model_save_path, exist_ok=True)
    early_stop = EarlyStopping(patience=3, verbose=True, path=os.path.join(model_save_path, "model_ldm_best_2dldm.pth"))

    val_loss = eval_upsampler_ldm_without_low_res(
        model=model,
        stage1=stage1,
        scheduler=scheduler,
        low_res_scheduler=low_res_scheduler,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        inferer=inferer

    )
    early_stop(val_loss, model)
    #run.log({'val_loss': val_loss})

    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    #torch.cuda.empty_cache()        
    
    for epoch in range(start_epoch, n_epochs):
        train_epoch_upsampler_ldm_without_low_res(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            low_res_scheduler=low_res_scheduler,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            inferer=inferer
            #run=run
        )
        #torch.cuda.empty_cache()

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_upsampler_ldm_without_low_res(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                low_res_scheduler=low_res_scheduler,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                inferer=inferer

            )
            #torch.cuda.empty_cache()
            early_stop(val_loss, model)
            #run.log({'val_loss': val_loss})

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
            
            if early_stop.early_stop:
                print("Early stopping")
                break
            
    print(f"Training finished!")
    print(f"Saving final model...")
    
    torch.save(raw_model.state_dict(), os.path.join(model_save_path, "model_ldm_2dldm.pth"))

    return val_loss

def train_epoch_upsampler_ldm_without_low_res(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    low_res_scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    inferer

) -> None:
    model.train()
    stage1.eval()
    epoch_loss = 0
    quant_batch = 0
    
    pbar = tqdm(enumerate(loader), total=len(loader), ncols=110)
    for step, x in pbar:
        images = x["image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            z_mu, z_sigma = stage1.encode(images)
            z = stage1.sampling(z_mu, z_sigma)
            noise = torch.randn_like(z).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noise_pred = inferer(
                inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, autoencoder_model=stage1
            )
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{(epoch_loss / (step + 1)):.4f}"
            }
        )
    #run.log({'loss': epoch_loss / (quant_batch + 1)})
    
def eval_upsampler_ldm_without_low_res(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    low_res_scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    inferer

) -> float:
    
    model.eval()
    stage1.eval()

    total_losses = OrderedDict()
    
    with torch.no_grad():
        for x in loader:
            images = x["image"].to(device)
                    
            with autocast(enabled=True):
                z_mu, z_sigma = stage1.encode(images)
                z = stage1.sampling(z_mu, z_sigma)

                noise = torch.randn_like(z).to(device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
                ).long()
                noise_pred = inferer(
                    inputs=images,
                    diffusion_model=model,
                    noise=noise,
                    timesteps=timesteps,
                    autoencoder_model=stage1,
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            losses = OrderedDict(loss=loss)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    return total_losses["loss"]
