from tqdm import tqdm
import torch


def train_vqvae(model, l1_loss,perceptual_loss, discriminator, adv_loss, adv_weight, perceptual_weight,
        train_loader, val_loader, optimizer_g, optimizer_d, n_epochs, eval_freq, device):
        
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    val_recon_epoch_loss_list = []

    for epoch in range(n_epochs):
        model.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer_g.zero_grad(set_to_none=True)

            # Generator part
            reconstruction, quantization_loss = model(images=images)
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)

            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

            epoch_loss += recons_loss.item()
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                }
            )
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

        if (epoch + 1) % eval_freq == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)

                    reconstruction, quantization_loss = model(images=images)

                    recons_loss = l1_loss(reconstruction.float(), images.float())

                    val_loss += recons_loss.item()

            val_loss /= val_step
            val_recon_epoch_loss_list.append(val_loss)

    return model