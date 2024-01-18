import torch
import torch.nn as nn
import numpy as np
from dataset import phasor_from_bpm
from models.unet import UNet
from models.unet_bt_conditioner import UNetBTConditioner
from models.encodec_model import WrappedEncodec

class UNetDiffusion(nn.Module):

    def __init__(self, n_channels = 128, alpha_steps = 100, time_emb_dim=16, start_channels_base2=6, 
                 n_layers=5, kernel_size_base2=1, n_groups=None,  inference_bpm=None, device='cpu'):
        super(UNetDiffusion, self).__init__()
        self.n_channels = n_channels
        self.inference_bpm = inference_bpm
        self.model = (           
            UNetBTConditioner(
                channels = n_channels, 
                time_emb_dim = time_emb_dim, 
                start_channels_base2 = start_channels_base2, 
                n_layers = n_layers, 
                kernel_size_base2 = kernel_size_base2, 
                n_groups = n_groups, device=device)
            if self.inference_bpm is not None else             
            UNet(
                channels = n_channels, 
                time_emb_dim = time_emb_dim, 
                start_channels_base2 = start_channels_base2, 
                n_layers = n_layers, 
                kernel_size_base2 = kernel_size_base2, 
                n_groups = n_groups, device=device))
        
        self.encodec = WrappedEncodec().to(device)
        self.alpha_steps = alpha_steps
        self._config_prior(torch.zeros(n_channels, 1, device=device), torch.ones(n_channels, 1, device=device))
        self.device = device

    def _config_prior(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, x, t, conditioner):
        return self.model(x, t, conditioner)
    
    def inference(self, x_0 = None, n_batch = 1, n_frames = 1024, T = None):
        if T is None or not isinstance(T, int):
            T = self.alpha_steps
        if x_0 is None:
            x_0 = self.sample(n_batch, n_frames)
        
        conditioner = None
        if self.inference_bpm is not None:
            conditioner = torch.from_numpy(np.array([phasor_from_bpm(self.inference_bpm) for _ in range(n_batch)]))[:,None,:].float().to(self.device)
        denoised_samples = [x_0]
        alpha = torch.arange(T, device=self.device)/T

        for t in range(1,T,1):
            a = alpha[t]*torch.ones(n_batch, 1, device=self.device)
            x_a = denoised_samples[-1] + (alpha[t] - alpha[t-1])*self.forward(denoised_samples[-1], a, conditioner)
            denoised_samples.append(x_a)
        
        return self.encodec.decode(denoised_samples[-1])
    
    def sample(self, n_batch, n_frames):
        return self.mean + self.std * torch.randn((n_batch, self.n_channels, n_frames), device=self.device)
