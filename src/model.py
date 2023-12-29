import torch
import torch.nn as nn
from models.unet import UNet

class UNetDiffusion(nn.Module):

    def __init__(self, n_channels = 128, alpha_steps = 1000):
        super(UNetDiffusion, self).__init__()
        self.n_channels = n_channels
        self.model = UNet(channels = n_channels)
        self.alpha_steps = alpha_steps
        self._config_prior(torch.zeros(n_channels), torch.ones(n_channels))

    def _config_prior(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, x, t):
        return self.model(x, t)
    
    def inference(self, n_batch = 1, T = None):
        if T is None or not isinstance(T, int):
            T = self.alpha_steps
        x_0 = self.sample(n_batch)
        denoised_samples = [x_0]
        alpha = torch.arange(T)/T
        for t in range(1,T,1):
            x_a = denoised_samples[-1] + (alpha[t] - alpha[t-1])*self.forward(denoised_samples[-1],alpha[t]*torch.ones(n_batch,1))
            denoised_samples.append(x_a)
        return denoised_samples
    
    def sample(self, n_batch):
        return self.mean + self.std * torch.randn((n_batch, self.n_channels))
