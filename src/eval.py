from src.models.UNet import UNet

from train import hyperparameters

if __name__ == "__main__":
    hparams = hyperparameters()
    model = UNet(1, 1, args=hparams)
