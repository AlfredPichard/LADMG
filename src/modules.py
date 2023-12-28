import numpy as np
import torch, torchvision, torchaudio
import torch.nn as nn


#############################################
"""
Low level - Encoding and Ada Group Normalization
"""
#############################################
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_steps = 10000):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.max_steps = max_steps

    def forward(self, alpha):
        pe = torch.zeros(self.max_steps, self.input_dim)
        position = torch.arange(0, self.max_steps).unsqueeze(1).float()
        omega = torch.exp((torch.arange(0, self.input_dim, 2, dtype=torch.float) * -(np.log(1000.0) / self.input_dim)))
        pe[:, 0::2] = torch.sin(position * omega)
        pe[:, 1::2] = torch.cos(position * omega)

        alpha = pe[:alpha.size(0)]
        return alpha
    

class AdaGN(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups=None):
        super(AdaGN, self).__init__()
        if n_groups is None:
            n_groups = in_channels

        self.groupnorm = nn.GroupNorm(n_groups, in_channels, affine=True, eps=1e-5)
        self.lin_time_proj = nn.Linear(time_emb_dim, 2*in_channels)
        self.in_channels = in_channels

    def forward(self, x, pos_enc):
        '''
        with: 
            D = input dimension
            C = in_channels
            T = time_emb_dim
            N = batch size

        x : (N, C, D)
        pos_enc : (N, T)
        t : (N, T) --> (N, 2 * C, D)
        t_a, t_b : (N, C, D)
        return : (N, C, D)
        '''
        t = self.lin_time_proj(pos_enc).unsqueeze(2).repeat(1,1,x.shape[-1])
        t_a, t_b = t[:,:self.in_channels,:], t[:,self.in_channels:,:]
        return t_a*self.groupnorm(x) + t_b



#############################################
"""
Higher level - Elementary blocks
"""
#############################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2):
        super(ConvBlock, self).__init__()            
        if not isinstance(kernel_size_base2,int) or not kernel_size_base2 > 0:
            kernel_size_base2 = 1

        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = AdaGN(out_channels, time_emb_dim, n_groups=n_groups)
        self.activation = nn.SiLU()

    def forward(self, x, pos_enc):
        x = self.conv(x)
        x = self.norm(x, pos_enc)
        return self.activation(x) + x


class BottomBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups, kernel_size_base2,  num_block=2, out_channels=None):
        super(BottomBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels * 2
        
        self.blocks = [ConvBlock(in_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2)]
        for _ in range(1, num_block):
            self.blocks.append(ConvBlock(out_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2))

        self.activation = nn.SiLU()

        if not isinstance(kernel_size_base2,int) or not kernel_size_base2 > 0:
            kernel_size_base2 = 1
        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.decoder = nn.ConvTranspose1d(out_channels, out_channels // 2, kernel_size, padding=padding, stride=2, output_padding=1)

    def forward(self, x, pos_enc):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc)
        return self.activation(self.decoder(x))


class OutBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups, kernel_size_base2, num_block=2, out_channels=None):
        super(OutBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        
        self.blocks = [ConvBlock(in_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2)]
        for _ in range(1, num_block):
            self.blocks.append(ConvBlock(out_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2))

        self.activation = nn.SiLU()

    def forward(self, x, pos_enc, skip):
        x = torch.cat((skip, x), dim=-2)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc)
        return x



#############################################
"""
Higher level - Encoding and Decoding blocks
"""
#############################################
class EncodeBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups, kernel_size_base2, block_size=2, out_channels=None):
        super(EncodeBlock, self).__init__()
        if out_channels is None:
            out_channels = 2 * in_channels
        
        self.blocks = [ConvBlock(in_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2)]
        for _ in range(1, block_size):
            self.blocks.append(ConvBlock(out_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2))

        self.activation = nn.SiLU()

        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.encoder = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, stride=2)
        self.skip = None

    def forward(self, x, pos_enc):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc)
        self.skip = x
        return self.activation(self.encoder(x))


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, num_block=2, kernel_size_base2=1, out_channels=None, n_groups=None):
        super(DecodeBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        
        self.blocks = []
        self.blocks.append(ConvBlock(in_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))
        for _ in range(1, num_block):
            self.blocks.append(ConvBlock(out_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))

        self.activation = nn.SiLU()

        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.decoder = nn.ConvTranspose1d(out_channels, out_channels // 2, kernel_size, padding=padding, stride=2, output_padding=1)

    def forward(self, x, pos_enc, skip):
        x = torch.cat((skip, x), dim=-2)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc)
        return self.activation(self.decoder(x))


