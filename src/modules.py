import torch, torchvision, torchaudio
import torch.nn as nn

class Positional_Encoding(nn.Module):
    
    def __init__(self, input_dim, max_steps = 2):
        super(Positional_Encoding, self).__init__()
        self.input_dim = input_dim
        self.max_steps = max_steps

    def sin_cos(self, a, i):
        if i:
            return torch.cos(a)
        else:
            return torch.sin(a)

    def forward(self, alpha):
        alpha = alpha.view(1, alpha.shape[0])
        alpha = [self.sin_cos(alpha/(self.max_steps**(2*(k//2)/self.input_dim)),k%2) for k in range(self.input_dim)]
        pos_enc = torch.cat(alpha)
        return torch.transpose(pos_enc,0,1)

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
    
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim, n_groups=None, kernel_size_base2 = None):
        super(ConvBlock, self).__init__()            
        if not isinstance(kernel_size_base2,int) or not kernel_size_base2 > 0:
            kernel_size_base2 = 1
        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = AdaGN(out_channels,time_emb_dim, n_groups=n_groups)
        self.activation = nn.SiLU()

    def forward(self, x, pos_enc):
        x = self.conv(x)
        x = self.norm(x, pos_enc)
        return self.activation(x) + x
    
class EncodeBlock(nn.Module):

    def __init__(self, in_channels, time_emb_dim, num_block = 2, out_channels=None, n_groups = None, kernel_size_base2 = None):
        super(EncodeBlock, self).__init__()
        if not isinstance(out_channels, int):
            out_channels = 2 * in_channels
        
        self.blocks = []
        self.blocks.append(ConvBlock(in_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))
        for _ in range(1, num_block):
            self.blocks.append(ConvBlock(out_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))

        self.activation = nn.SiLU()

        if not isinstance(kernel_size_base2,int) or not kernel_size_base2 > 0:
            kernel_size_base2 = 1
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

    def __init__(self, in_channels, time_emb_dim, num_block = 2, out_channels=None, n_groups = None, kernel_size_base2 = None):
        super(DecodeBlock, self).__init__()
        if not isinstance(out_channels, int):
            out_channels = in_channels // 2
        
        self.blocks = []
        self.blocks.append(ConvBlock(in_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))
        for _ in range(1, num_block):
            self.blocks.append(ConvBlock(out_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))

        self.activation = nn.SiLU()

        if not isinstance(kernel_size_base2,int) or not kernel_size_base2 > 0:
            kernel_size_base2 = 1
        kernel_size = 2**kernel_size_base2 + 1
        padding = 2**(kernel_size_base2 - 1)

        self.decoder = nn.ConvTranspose1d(out_channels, out_channels // 2, kernel_size, padding=padding, stride=2, output_padding=1)

    def forward(self, x, pos_enc, skip):
        x = torch.cat((skip, x), dim=-2)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc)
        return self.activation(self.decoder(x))
    
class BottomBlock(nn.Module):

    def __init__(self, in_channels, time_emb_dim, num_block = 2, out_channels=None, n_groups = None, kernel_size_base2 = None):
        super(BottomBlock, self).__init__()
        if not isinstance(out_channels, int):
            out_channels = in_channels * 2
        
        self.blocks = []
        self.blocks.append(ConvBlock(in_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))
        for _ in range(1, num_block):
            self.blocks.append(ConvBlock(out_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))

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

    def __init__(self, in_channels, time_emb_dim, num_block = 2, out_channels=None, n_groups = None, kernel_size_base2 = None):
        super(OutBlock, self).__init__()
        if not isinstance(out_channels, int):
            out_channels = in_channels // 2
        
        self.blocks = []
        self.blocks.append(ConvBlock(in_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))
        for _ in range(1, num_block):
            self.blocks.append(ConvBlock(out_channels, out_channels, time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))

        self.activation = nn.SiLU()

    def forward(self, x, pos_enc, skip):
        x = torch.cat((skip, x), dim=-2)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, pos_enc)
        return x

    
class UNet(nn.Module):

    def __init__(self, channels = 2, time_emb_dim = 16, start_channels_base2 = 6, n_layers = 5, n_groups = None, kernel_size_base2 = None):
        super(UNet, self).__init__()
        self.n_layers = n_layers
        self.encode = []
        self.decode = []
        self.encode.append(EncodeBlock(channels, time_emb_dim, out_channels = 2**start_channels_base2, n_groups=n_groups, kernel_size_base2=kernel_size_base2))
        self.decode.append(OutBlock(2**(start_channels_base2 + 1), time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))
        for i in range(n_layers - 2):
            self.encode.append(EncodeBlock(2**(start_channels_base2 + i), time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))
            self.decode.append(DecodeBlock(2**(start_channels_base2 + i + 2), time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2))
        self.bottom = BottomBlock(2**(start_channels_base2 + n_layers - 2), time_emb_dim, n_groups=n_groups, kernel_size_base2=kernel_size_base2)

        self.last_conv = nn.Conv1d(2**(start_channels_base2), channels, 1)
        self.pos_encoding = Positional_Encoding(time_emb_dim)
        
    def forward(self, x, t):
        pos_enc = self.pos_encoding(t)
        for i in range(self.n_layers - 1):
            x = self.encode[i](x, pos_enc)
        x = self.bottom(x, pos_enc)
        for i in range(self.n_layers - 2, -1, -1):
            x = self.decode[i](x, pos_enc, self.encode[i].skip)
        return self.last_conv(x)


if __name__ == "__main__":

    model = UNet()
    x = torch.randn((16, 2, 64))
    t = torch.rand((16))
    y = model(x, t)
    print(y.shape)