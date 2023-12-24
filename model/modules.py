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

class AdaGN(nn.module):

    def __init__(self, n_channels, n_groups, time_emb_dim):
        super(AdaGN, self).__init__()
        self.groupnorm = nn.GroupNorm(n_groups, n_channels)
        self.lin_time_proj = nn.Linear(time_emb_dim, 2)

    def forward(self, x, pos_enc):
        y = self.lin_time_proj(pos_enc)
        return y[:,:,0]*x + y[:,:,1]
    
class UNet(nn.module):

    def __init__(self, in_dim):
        super(UNet, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        return x