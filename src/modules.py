import numpy as np
import torch, torchvision, torchaudio
import torch.nn as nn
from models.clap import CLAP

#############################################
"""
Low level - Encoding and Ada Group Normalization
"""
#############################################
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_steps = 5000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.max_steps = max_steps
        self.omega = torch.exp((torch.arange(0, self.input_dim, 2, dtype=torch.float, device=self.device) * -(np.log(1000.0) / self.input_dim)))

    def forward(self, x):
        x = x.squeeze(1)
        pe = torch.zeros(x.size(0), self.input_dim, device=self.device)
        pe[:, 0::2] = torch.sin(100 * x * self.omega)
        pe[:, 1::2] = torch.cos(100 * x * self.omega)
        pe = pe[:x.size(0)]
        
        return pe
    
class ConvCLAP(nn.Module):
    def __init__(self, in_channels = None, out_channels = 32, same_dim = False, device='cpu'):
        super(ConvCLAP, self).__init__()
        self.device = device
        self.out_channels = out_channels
        if in_channels:
            self.in_channels = in_channels
        else:
            self.in_channels = self.out_channels
        self.conv_in = nn.Conv1d(self.in_channels, self.out_channels, 3, padding='same')
        if same_dim:
            self.conv_out = nn.Conv1d(self.out_channels, self.out_channels, 3, padding='same')
        else:
            self.conv_out = nn.Conv1d(self.out_channels, self.out_channels, 3, padding=1, stride=2)
        self.activation = nn.SiLU()

    def forward(self, embedding):
        return self.activation(self.conv_out(self.conv_in(embedding)))

class AdaGN(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups=None, dropout_p = 0.5, device='cpu'):
        super(AdaGN, self).__init__()
        if n_groups is None:
            n_groups = in_channels
        self.device = device

        self.p = dropout_p
        self.groupnorm = nn.GroupNorm(n_groups, in_channels, affine=True, eps=1e-5, device=self.device)
        self.lin_time_proj = nn.Linear(time_emb_dim, 2*in_channels, device=self.device)
        self.z_proj = nn.Linear(CLAP.CLAP_DIM, in_channels, device=self.device)
        self.batch_dropout = nn.Dropout(p = self.p)
        self.in_channels = in_channels
        self.activation = nn.SiLU()

    def forward(self, x, pos_enc, z_cond):
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
        t = self.activation(self.lin_time_proj(pos_enc).unsqueeze(2).repeat(1,1,x.shape[-1]))
        t_a, t_b = t[:,:self.in_channels,:], t[:,self.in_channels:,:]
        if z_cond is None:
            z_cond = torch.zeros((x.shape[0], CLAP.CLAP_DIM), device=self.device)
        else:
            batch_mask = torch.rand(x.shape[0], device=self.device)
            batch_mask =(batch_mask > self.p).int()/(1 - self.p)
            batch_mask = batch_mask.unsqueeze(1).repeat(1, CLAP.CLAP_DIM)
            z_cond = batch_mask * z_cond
            '''
            batch_mask = torch.unsqueeze(self.batch_dropout(torch.ones(z_cond.shape[0], requires_grad=False, device=self.device)), 1)
            print(batch_mask)
            z_cond = batch_mask * z_cond
            print(z_cond)
            '''
        z_cond = self.activation(self.z_proj(z_cond)).unsqueeze(2).repeat(1,1,x.shape[-1])
        #z_cond = self.activation(self.z_proj(z_cond)).unsqueeze(2).repeat(1,1,x.shape[-1])
        out =  z_cond * (t_a*self.groupnorm(x) + t_b)
        #print(out[:2, :4, :4])
        return out



#############################################
"""
Higher level - Elementary blocks
"""
#############################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2, dropout_p = 0.2, device='cpu'):
        super(ConvBlock, self).__init__()            
        if not isinstance(kernel_size_base2,int) or not kernel_size_base2 > 0:
            kernel_size_base2 = 1

        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)
        self.device = device

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, device=self.device)
        self.norm = AdaGN(out_channels, time_emb_dim, n_groups=n_groups, device=self.device)
        self.activation = nn.SiLU()
        if not dropout_p:
            dropout_p = 0
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, pos_enc, z_cond):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.norm(x, pos_enc, z_cond)
        return self.activation(x) + x


class BottomBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups, kernel_size_base2,  num_block=2, out_channels=None, device='cpu'):
        super(BottomBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels * 2
        self.device = device

        self.blocks = nn.ModuleList(
            [ConvBlock(in_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2, device=self.device),
            *[ConvBlock(out_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2, device=self.device) for _ in range(num_block-1)]]
        )
        self.activation = nn.SiLU()

        if not isinstance(kernel_size_base2,int) or not kernel_size_base2 > 0:
            kernel_size_base2 = 1
        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.decoder = nn.ConvTranspose1d(out_channels, out_channels // 2, kernel_size, padding=padding, stride=2, output_padding=1, device=self.device)

    def forward(self, x, pos_enc, z_cond):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc, z_cond)
        return self.activation(self.decoder(x))


class OutBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups, kernel_size_base2, num_block=2, out_channels=None, device='cpu'):
        super(OutBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        self.device = device

        self.blocks = nn.ModuleList(
            [ConvBlock(in_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2, device=self.device),
            *[ConvBlock(out_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2, device=self.device) for _ in range(num_block-1)]]
        )
        self.activation = nn.SiLU()

    def forward(self, x, pos_enc, skip, z_cond):
        x = torch.cat((skip, x), dim=-2)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc, z_cond)
        return x



#############################################
"""
Higher level - Encoding and Decoding blocks
"""
#############################################
class EncodeBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, n_groups, kernel_size_base2, block_size=2, out_channels=None, device='cpu'):
        super(EncodeBlock, self).__init__()
        if out_channels is None:
            out_channels = 2 * in_channels
        self.device = device

        self.blocks = nn.ModuleList(
            [ConvBlock(in_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2, device=self.device),
            *[ConvBlock(out_channels, out_channels, time_emb_dim, n_groups, kernel_size_base2, device=self.device) for _ in range(block_size-1)]]
        )
        self.activation = nn.SiLU()

        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.encoder = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, stride=2, device=self.device)

    def forward(self, x, pos_enc, z_cond):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc, z_cond)
        return self.activation(self.encoder(x)), x


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, num_block=2, kernel_size_base2=1, out_channels=None, n_groups=None, device = 'cpu'):
        super(DecodeBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        self.device = device

        self.blocks = nn.ModuleList(
            [ConvBlock(in_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2, device=self.device),
            *[ConvBlock(out_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2, device=self.device) for _ in range(num_block-1)]]
        )
        self.activation = nn.SiLU()

        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.decoder = nn.ConvTranspose1d(out_channels, out_channels // 2, kernel_size, padding=padding, stride=2, output_padding=1, device=self.device)

    def forward(self, x, pos_enc, skip, z_cond):
        x = torch.cat((skip, x), dim=-2)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc, z_cond)
        return self.activation(self.decoder(x))


