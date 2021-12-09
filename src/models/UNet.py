from abc import ABC

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.transforms.functional as tv_func
from torch.optim import Adam


# noinspection PyTypeChecker
class DoubleConv(pl.core.LightningModule, ABC):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# noinspection PyTypeChecker,PyCallingNonCallable
class UNet(pl.core.LightningModule, ABC):
    def __init__(self, in_channels=1, out_channels=3, features=None, batch_size=32, learning_rate=1e-3):
        super(UNet, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_hyperparameters()

        if features is None:
            features = [64, 128]

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels = out_channels

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = func.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = func.binary_cross_entropy_with_logits(logits, y)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = func.binary_cross_entropy_with_logits(logits, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)  # Conv2D
            skip_connections.append(x)
            x = self.pool(x)  # MaxPooling

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose
            # //2 because we have a step of 2
            skip_connection = skip_connections[idx // 2]

            # Verify that the skip connections have the same shape
            if x.shape != skip_connection.shape:
                x = tv_func.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # DoubleConv

        if self.out_channels == 1:
            # Do not forget to use a loss that combines a Sigmoid
            return self.final_conv(x)

        return nn.Softmax2d()(self.final_conv(x))
