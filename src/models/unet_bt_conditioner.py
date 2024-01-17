import torch
import torch.nn as nn

from modules import PositionalEncoding
from modules_light import EncodeBlock, DecodeBlock, BottomBlock, OutBlock, ConditionerBlock


class UNetBTConditioner(nn.Module):

    def __init__(self, channels=2, time_emb_dim=64, start_channels_base2=6, n_layers=5, kernel_size_base2=1, n_groups=None, device='cpu'):
        super(UNetBTConditioner, self).__init__()
        self.n_layers = n_layers

        self.encode = nn.ModuleList([
            EncodeBlock(
                in_channels=channels, 
                out_channels = 2**start_channels_base2,
                time_emb_dim=time_emb_dim, 
                n_groups=n_groups, 
                kernel_size_base2=kernel_size_base2, 
                use_bt_conditioner=True,
                device = device),
            *[EncodeBlock(
                in_channels=2**(start_channels_base2 + i), 
                time_emb_dim=time_emb_dim, 
                n_groups=n_groups, 
                kernel_size_base2=kernel_size_base2,
                use_bt_conditioner=True,
                device = device) for i in range(n_layers - 2)]
            ])
        
        self.bottom = BottomBlock(
            in_channels=2**(start_channels_base2 + n_layers - 2), 
            time_emb_dim=time_emb_dim, 
            n_groups=n_groups, 
            kernel_size_base2=kernel_size_base2,
            device = device)
        
        self.decode = nn.ModuleList([
            OutBlock(
                in_channels=2**(start_channels_base2 + 1), 
                time_emb_dim=time_emb_dim, 
                n_groups=n_groups, 
                kernel_size_base2=kernel_size_base2, 
                use_bt_conditioner=True,
                device = device),
            *[DecodeBlock(
                in_channels=2**(start_channels_base2 + i + 2),
                time_emb_dim=time_emb_dim, 
                n_groups=n_groups,
                kernel_size_base2=kernel_size_base2,
                use_bt_conditioner=True,
                device = device) for i in range(n_layers - 2)]
            ])
        
        self.conditioner = nn.ModuleList([
            *[ConditionerBlock(
                in_channels=1, 
                hidden_channels=2**(start_channels_base2 + i), 
                stride=2**i,
                kernel_size_base2=kernel_size_base2,
                device=device) for i in range(n_layers-1)]
            ])

        self.last_conv = nn.Conv1d(2**(start_channels_base2), channels, 1, device=device)
        self.pos_encoding = PositionalEncoding(time_emb_dim, device = device)
        
    def forward(self, x, t, y):
        pos_enc = self.pos_encoding(t)
        skip_connections = []
        bt_embeddings = []

        for i in range(self.n_layers - 1):
            bt_embedding = self.conditioner[i](y)
            x, skip = self.encode[i](x, pos_enc, bt_embedding)
            bt_embeddings.append(bt_embedding)
            skip_connections.append(skip)

        x = self.bottom(x, pos_enc)

        for i in range(self.n_layers - 2, -1, -1):            
            x = self.decode[i](x, pos_enc, skip_connections.pop(), bt_embeddings.pop())

        return self.last_conv(x)
