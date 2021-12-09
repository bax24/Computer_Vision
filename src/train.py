import os

import numpy as np
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.SegmentationDataset import SegmentationDataset
from src.config import DATASET_ROOT, PROJECT_ROOT, WANDB_KEY
from src.models.UNet import UNet

DATASET_PATH = os.path.join(DATASET_ROOT, "clean")
SEED = 15
TEST_SIZE = 0.33
SIZE = (64, 64)
FEATURES = [64, 128, 256]
BATCH_SIZE = 128


def get_transforms(size):
    def transform(img, mask):
        img = np.resize(img, size)
        img = torch.tensor(np.array([img]), dtype=torch.float16)

        mask = np.resize(mask.astype(np.uint8), size)
        mask = torch.tensor(np.array([mask]), dtype=torch.float16)
        return img, mask

    return transform


checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",
    dirpath=os.path.join(PROJECT_ROOT, "models/"),
    filename="sample-v2-{epoch:02d}-{val_loss:.5f}",
    save_top_k=3,
    mode="min"
)

seed_everything(SEED, workers=True)
with open(WANDB_KEY, "r") as wandb_key:
    wandb.login(key=wandb_key.read()[:-1])

wandb_logger = WandbLogger(project="Computer vision", log_model="all")


train_dataset = SegmentationDataset(DATASET_PATH, train=True, test_size=TEST_SIZE,
                                    transforms=get_transforms(SIZE))
valid_dataset = SegmentationDataset(DATASET_PATH, train=False, test_size=TEST_SIZE,
                                    transforms=get_transforms(SIZE))

model = UNet(1, 1, features=FEATURES)
train_dataloader = DataLoader(train_dataset, num_workers=3, batch_size=BATCH_SIZE)
valid_dataloader = DataLoader(valid_dataset, num_workers=3, batch_size=BATCH_SIZE)

wandb_logger.watch(model)
trainer = Trainer(gpus=1,
                  precision=16,
                  check_val_every_n_epoch=1,
                  auto_lr_find=1e-3,
                  max_epochs=3,
                  logger=wandb_logger,
                  callbacks=[checkpoint_callback])

# trainer.tune(model)
trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)

model.load_from_checkpoint(checkpoint_callback.best_model_path)
trainer.validate(model, val_dataloaders=valid_dataloader)
