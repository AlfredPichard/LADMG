import torch

from models.unet import UNet


if __name__ == "__main__":

    model = UNet()
    x = torch.randn((16, 2, 64))
    t = torch.rand((16))
    y = model(x, t)
    print(y.shape)