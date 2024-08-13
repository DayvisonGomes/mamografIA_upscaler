import torch
from tqdm import trange

L1Loss = torch.nn.L1Loss(reduction="sum")


def loss_function(recon_x, x, mu, log_var, beta):
    bce = L1Loss(recon_x, x)
    kld = -0.5 * beta * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld


def train_vae(model, n_epochs, beta, train_loader,val_loader, optimizer,device):
    
    avg_train_losses = []
    test_losses = []

    t = trange(n_epochs, leave=True, desc="epoch 0, average train loss: ?, test loss: ?")
    for epoch in t:
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            optimizer.zero_grad()
    
            recon_batch, mu, log_var, _ = model(inputs)
            loss = loss_function(recon_batch, inputs, mu, log_var, beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_losses.append(epoch_loss / len(train_loader.dataset))

        # Test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(device)
                recon, mu, log_var, _ = model(inputs)
                # sum up batch loss
                test_loss += loss_function(recon, inputs, mu, log_var, beta).item()
        test_losses.append(test_loss / len(val_loader.dataset))

        t.set_description(  # noqa: B038
            f"epoch {epoch + 1}, average train loss: " f"{avg_train_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}"
        )
    return model, avg_train_losses, test_losses