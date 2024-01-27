import numpy as np
import torch, torchvision, torchaudio
import torch.nn as nn

from modules import ConvBlock


"""
Temporary file with simpler blocks for BT Conditioning
"""


#############################################
"""
Mid-High level - Elementary blocks
"""
#############################################
class OutBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups, kernel_size_base2, out_channels=None, use_bt_conditioner=False, device='cpu'):
        super(OutBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        self.device = device
        self.bt_adapted_channels = out_channels if not use_bt_conditioner else 2*out_channels

        self.conv_block_1 = ConvBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            time_emb_dim=time_emb_dim, 
            n_groups=n_groups, 
            kernel_size_base2=kernel_size_base2, 
            device=self.device)
        
        self.conv_block_2 = ConvBlock(
            in_channels=self.bt_adapted_channels, 
            out_channels=out_channels, 
            time_emb_dim=time_emb_dim, 
            n_groups=n_groups, 
            kernel_size_base2=kernel_size_base2, 
            device=self.device)
        
        self.activation = nn.SiLU()

    def forward(self, x, pos_enc, skip, bt_embedding=None):
        x = torch.cat((skip, x), dim=-2)
        x = self.conv_block_1(x, pos_enc)
        if bt_embedding is not None:
            x = torch.cat((x, bt_embedding), dim=1)
        x = self.conv_block_2(x, pos_enc)
        return x


class ConditionerBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, stride, kernel_size_base2, device='cpu'):
        super(ConditionerBlock, self).__init__()
        self.device = device

        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size_base2, device=self.device),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, stride=stride, kernel_size=kernel_size_base2, device=self.device),
            nn.SiLU()
        )

    def forward(self, x):
        return self.blocks(x)


#############################################
"""
Highest level - Encoding, Bottom and Decoding blocks
"""
#############################################
class EncodeBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups, kernel_size_base2, out_channels=None, use_bt_conditioner=False, device='cpu'):
        super(EncodeBlock, self).__init__()
        if out_channels is None:
            out_channels = 2 * in_channels

        self.bt_adapted_channels = out_channels if not use_bt_conditioner else 2*out_channels
        self.device = device

        self.conv_block_1 = ConvBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            time_emb_dim=time_emb_dim, 
            n_groups=n_groups, 
            kernel_size_base2=kernel_size_base2,
            device=self.device)
        
        self.conv_block_2 = ConvBlock(
            in_channels=self.bt_adapted_channels, 
            out_channels=out_channels, 
            time_emb_dim=time_emb_dim, 
            n_groups=n_groups, 
            kernel_size_base2=kernel_size_base2, 
            device=self.device)

        self.activation = nn.SiLU()

        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.encoder = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, stride=2, device=self.device)

    def forward(self, x, pos_enc, bt_embedding=None):
        x = self.conv_block_1(x, pos_enc)
        if bt_embedding is not None:
            x = torch.cat((x, bt_embedding), dim=1)
        x = self.conv_block_2(x, pos_enc)
        return self.activation(self.encoder(x)), x


class BottomBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups, kernel_size_base2, out_channels=None, device='cpu'):
        super(BottomBlock, self).__init__()
        if not isinstance(kernel_size_base2,int) or not kernel_size_base2 > 0:
            kernel_size_base2 = 1
        if out_channels is None:
            out_channels = in_channels * 2
        
        self.device = device

        self.conv_block = ConvBlock(in_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2, device=self.device)
        self.activation = nn.SiLU()

        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.decoder = nn.ConvTranspose1d(out_channels, out_channels // 2, kernel_size, padding=padding, stride=2, output_padding=1, device=self.device)

    def forward(self, x, pos_enc, bt_embedding=None):
        if bt_embedding is not None:
            x = torch.cat((x, bt_embedding), dim=1)
        x = self.conv_block(x, pos_enc)
        return self.activation(self.decoder(x))


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, kernel_size_base2=1, out_channels=None, n_groups=None, use_bt_conditioner=False, device = 'cpu'):
        super(DecodeBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        self.device = device
        self.bt_adapted_channels = out_channels if not use_bt_conditioner else 2*out_channels

        self.conv_block_1 = ConvBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            time_emb_dim=time_emb_dim, 
            n_groups=n_groups, 
            kernel_size_base2=kernel_size_base2, 
            device=self.device)
        
        self.conv_block_2 = ConvBlock(
            in_channels=self.bt_adapted_channels, 
            out_channels=out_channels, 
            time_emb_dim=time_emb_dim, 
            n_groups=n_groups, 
            kernel_size_base2=kernel_size_base2, 
            device=self.device)

        self.activation = nn.SiLU()

        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.decoder = nn.ConvTranspose1d(out_channels, out_channels // 2, kernel_size, padding=padding, stride=2, output_padding=1, device=self.device)

    def forward(self, x, pos_enc, skip, bt_embedding=None):
        x = torch.cat((skip, x), dim=-2)
        x = self.conv_block_1(x, pos_enc)
        if bt_embedding is not None:
            x = torch.cat((x, bt_embedding), dim=1)
        x = self.conv_block_2(x, pos_enc)
        return self.activation(self.decoder(x))


