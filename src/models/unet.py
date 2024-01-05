import torch
import torch.nn as nn

from modules import EncodeBlock, DecodeBlock, BottomBlock, OutBlock, PositionalEncoding


class UNet(nn.Module):

    def __init__(self, channels=2, time_emb_dim=16, start_channels_base2=6, n_layers=5, kernel_size_base2=1, n_groups=None, device='cpu'):
        super(UNet, self).__init__()
        self.n_layers = n_layers

        self.encode = nn.ModuleList([
            EncodeBlock(
                in_channels=channels, 
                out_channels = 2**start_channels_base2,
                time_emb_dim=time_emb_dim, 
                n_groups=n_groups, 
                kernel_size_base2=kernel_size_base2, 
                device = device)
            ])
        
        self.decode = nn.ModuleList([
            OutBlock(
                in_channels=2**(start_channels_base2 + 1), 
                time_emb_dim=time_emb_dim, 
                n_groups=n_groups, 
                kernel_size_base2=kernel_size_base2, 
                device = device)
            ])

        for i in range(n_layers - 2):
            self.encode.append(
                EncodeBlock(
                    in_channels=2**(start_channels_base2 + i), 
                    time_emb_dim=time_emb_dim, 
                    n_groups=n_groups, 
                    kernel_size_base2=kernel_size_base2, 
                    device = device)
                )
            self.decode.append(
                DecodeBlock(
                    in_channels=2**(start_channels_base2 + i + 2), 
                    time_emb_dim=time_emb_dim, 
                    n_groups=n_groups,
                    kernel_size_base2=kernel_size_base2,
                    device = device)
                )
            
        self.bottom = BottomBlock(
            in_channels=2**(start_channels_base2 + n_layers - 2), 
            time_emb_dim=time_emb_dim, 
            n_groups=n_groups, 
            kernel_size_base2=kernel_size_base2,
            device = device)

        self.last_conv = nn.Conv1d(2**(start_channels_base2), channels, 1, device=device)
        self.pos_encoding = PositionalEncoding(time_emb_dim, device = device)
        
    def forward(self, x, t):
        pos_enc = self.pos_encoding(t)
        for i in range(self.n_layers - 1):
            x = self.encode[i](x, pos_enc)
        x = self.bottom(x, pos_enc)
        for i in range(self.n_layers - 2, -1, -1):
            x = self.decode[i](x, pos_enc, self.encode[i].skip)
        return self.last_conv(x)
