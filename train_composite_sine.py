from pathlib import Path
import argparse
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from data_generation import CompositeSineDataset
from experiment_config import copy_config_to_run, load_config, resolve_checkpoint_dir, resolve_run_dir
from stack import doubleStack


class CompositeSineRegressor(L.LightningModule):
    def __init__(self, neurons_per_layer, lr=1e-3, weight_decay=1e-3, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = doubleStack(neurons_per_layer, dropout=dropout)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        # expect x shape (batch, 1)
        return self.model(x)

    def shared_step(self, batch, mode):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a double stack regressor on composite sine data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/composite_sine_plus_bottleneck1.yaml"),
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def auto_num_workers():
    return min(4, max(1, (os.cpu_count() or 2) - 1))


def build_loader(dataset, batch_size, shuffle, dataloader_config):
    num_workers = dataloader_config["num_workers"]
    if num_workers == "auto":
        num_workers = auto_num_workers()

    pin_memory = dataloader_config["pin_memory"]
    if pin_memory == "auto":
        pin_memory = torch.cuda.is_available()

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }

    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(dataloader_config["persistent_workers"])
        kwargs["prefetch_factor"] = int(dataloader_config["prefetch_factor"])

    return DataLoader(dataset, **kwargs)


def resolve_trainer_value(value, auto_value):
    return auto_value if value == "auto" else value


def build_callbacks(config):
    trainer_config = config["trainer"]
    if not trainer_config["enable_checkpointing"]:
        return []

    checkpoint_config = config["checkpoint"]
    return [
        ModelCheckpoint(
            dirpath=resolve_checkpoint_dir(config),
            filename=checkpoint_config["filename"],
            monitor=checkpoint_config["monitor"],
            mode=checkpoint_config["mode"],
            save_top_k=checkpoint_config["save_top_k"],
            save_last=checkpoint_config["save_last"],
            every_n_epochs=checkpoint_config["every_n_epochs"],
        )
    ]


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    run_dir = resolve_run_dir(config)
    copy_config_to_run(args.config, run_dir)
    L.seed_everything(config["seed"], workers=True)

    dataset_config = config["dataset"]
    train_ds = CompositeSineDataset(
        n_samples=dataset_config["n_train"],
        function=dataset_config["function"],
        p=dataset_config["p"],
        interval=tuple(dataset_config["interval"]),
        noise_std=dataset_config["noise_std"],
        seed=dataset_config["train_seed"],
    )
    val_ds = CompositeSineDataset(
        n_samples=dataset_config["n_val"],
        function=dataset_config["function"],
        p=dataset_config["p"],
        interval=tuple(dataset_config["interval"]),
        noise_std=dataset_config["noise_std"],
        seed=dataset_config["val_seed"],
    )

    sample_y = train_ds.y[:5]
    output_dim = sample_y.shape[1]

    model_config = config["model"]
    neurons_per_layer = [
        model_config["input_dim"],
        model_config["hidden_dim_1"],
        model_config["bottleneck_dim"],
        model_config["hidden_dim_2"],
        output_dim,
    ]

    dataloader_config = config["dataloader"]
    batch_size = dataloader_config["batch_size"] or dataset_config["n_train"]
    train_loader = build_loader(train_ds, batch_size, shuffle=True, dataloader_config=dataloader_config)
    val_loader = build_loader(val_ds, batch_size, shuffle=False, dataloader_config=dataloader_config)

    optimizer_config = config["optimizer"]
    model = CompositeSineRegressor(
        neurons_per_layer,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
        dropout=model_config["dropout"],
    )

    trainer_config = config["trainer"]
    callbacks = build_callbacks(config)
    accelerator = resolve_trainer_value(
        trainer_config["accelerator"],
        "gpu" if torch.cuda.is_available() else "cpu",
    )
    devices = resolve_trainer_value(trainer_config["devices"], 1)
    trainer = L.Trainer(
        max_epochs=trainer_config["max_epochs"],
        check_val_every_n_epoch=trainer_config["check_val_every_n_epoch"],
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=trainer_config["enable_checkpointing"],
        log_every_n_steps=trainer_config["log_every_n_steps"],
        callbacks=callbacks,
        default_root_dir=run_dir,
    )
    trainer.fit(model, train_loader, val_loader)
    print("Model training finished.")
