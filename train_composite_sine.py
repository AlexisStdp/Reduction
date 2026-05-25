from stack import doubleStack, Stack, FFN

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lightning as L

from data_generation import CompositeSineDataset
import os


class CompositeSineRegressor(L.LightningModule):
    def __init__(self, neurons_per_layer, lr=1e-3):
        super().__init__()
        self.model = doubleStack(neurons_per_layer)
        self.lr = lr

    def forward(self, x):
        # expect x shape (batch, 1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)


if __name__ == '__main__':
    # dataset / training params
    n_train = 256
    n_val = 16
    batch_size = n_train
    function = 'compositeSinePlus' #  'compositeSine', 'compositeSinePlus'
    p = [0.4, 1.0, 2.0, -3.0]
    noise_std = 0.05

    # build datasets
    train_ds = CompositeSineDataset(n_samples=n_train, function=function, p=p, noise_std=noise_std, seed=42)
    val_ds = CompositeSineDataset(n_samples=n_val, function=function, p=p, noise_std=noise_std, seed=43)

    # inspect output dimension to configure network
    sample_y = train_ds.y[:5]
    output_dim = sample_y.shape[1]

    neurons_per_layer = [1, 1024, 1, 1024, output_dim]

    num_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = CompositeSineRegressor(neurons_per_layer, lr=1e-3)

    # reduce overhead: run on GPU if available, don't save checkpoints each epoch,
    # and validate only every 50 epochs to avoid long validation/checkpoint stalls
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if torch.cuda.is_available() else None
    trainer = L.Trainer(max_epochs=2000, check_val_every_n_epoch=50, accelerator=accelerator, devices=devices)
    trainer.fit(model, train_loader, val_loader)
    print("Model training finished.")