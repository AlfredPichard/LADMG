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
    
    def inference(self, x_0 = None, n_batch = 1, n_frames = 100, T = None):
        if T is None or not isinstance(T, int):
            T = self.alpha_steps
        if x_0 is None:
            x_0 = self.sample(n_batch, n_frames)
        denoised_samples = [x_0]
        alpha = torch.arange(T)/T
        for t in range(1,T,1):
            x_a = denoised_samples[-1] + (alpha[t] - alpha[t-1])*self.forward(denoised_samples[-1],alpha[t]*torch.ones(n_batch,1))
            denoised_samples.append(x_a)
        return denoised_samples
    
    def sample(self, n_batch, n_frames):
        return self.mean + self.std * torch.randn((n_batch, self.n_channels, n_frames))
    
    def interpolate(self, x_0 = None, x_1 = None, n_batch = 1, n_inter = 10, n_frames = 100, T = None):
        if x_0 is None:
            x_0 = self.sample(n_batch, n_frames)
        if x_1 is None:
            x_1 = self.sample(n_batch, n_frames)

        interpolations = []
        for i in range(n_inter + 1):
            a = i/n_inter
            x_inter = a * x_0 + (1 - a) * x_1
            interpolations.append(self.inference(x_0 = x_inter, n_batch=n_batch, n_frames=n_frames, T=T))
        return interpolations

