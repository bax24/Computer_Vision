import os

import numpy as np
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.config import DATASET_ROOT, PROJECT_ROOT, WANDB_KEY
from src.models.UNet import UNet


def get_transforms(size):
    def transform(img, mask):
        img = np.resize(img, size)
        img = torch.tensor(np.array([img]), dtype=torch.float16)

        mask = np.resize(mask.astype(np.uint8), size)
        mask = torch.tensor(np.array([mask]), dtype=torch.float16)
        return img, mask

    return transform


def get_checkpoint_callback():
    return ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(PROJECT_ROOT, "models/"),
        filename="features-128-256-512-{epoch:02d}-{val_loss:.5f}",
        save_top_k=1,
        mode="min"
    )


def init_training():
    seed_everything(hparams["seed"], workers=True)
    with open(WANDB_KEY, "r") as wandb_key:
        wandb.login(key=wandb_key.read()[:-1])


def hyperparameters():
    input_size = (128, 128)
    return {
        "dataset_path": os.path.join(DATASET_ROOT, "clean"),
        "seed": 15,
        "test_split": 0.33,
        "input_size": input_size,
        "model_features": [128, 256, 512],
        "batch_size": 128,
        "num_worker": 3,
        "transforms": get_transforms(input_size)
    }


if __name__ == "__main__":
    hparams = hyperparameters()
    init_training()

    model = UNet(1, 1, args=hparams)
    wandb_logger = WandbLogger(project="Computer vision", log_model="all")
    wandb_logger.watch(model)
    trainer = Trainer(
        gpus=1,
        precision=16,
        check_val_every_n_epoch=1,
        auto_lr_find=True,
        max_epochs=5,
        logger=wandb_logger,
        callbacks=[get_checkpoint_callback()],
    )

    trainer.tune(model)
    trainer.fit(model)
