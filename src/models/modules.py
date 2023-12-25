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

    def __init__(self, latent_dim, pos_enc_dim, n_channels = 1, n_groups = 6):
        super(UNet, self).__init__()
        self.in_dim = latent_dim
        self.norm = AdaGN(n_channels, n_groups)
        self.PE = Positional_Encoding(pos_enc_dim)

        #TOP U-LAYER
        self.conv_1_1 = nn.Conv1d(1, 64, 3)
        self.conv_1_2 = nn.Conv1d(64, 64, 3)
        self.conv_1_3 = nn.Conv1d(128, 64, 3)
        self.conv_1_4 = nn.Conv1d(64, 64, 3)
        self.conv1x1 = nn.Conv1d(64, 2, 1)

        #SECOND U-LAYER
        self.conv_2_1 = nn.Conv1d(64, 128, 3)
        self.conv_2_2 = nn.Conv1d(128, 128, 3)
        self.conv_2_3 = nn.Conv1d(256, 128, 3)
        self.conv_2_4 = nn.Conv1d(128, 128, 3)
        self.upconv_2 = nn.ConvTranspose1d(128, 64, 2)

        #THIRD U-LAYER
        self.conv_3_1 = nn.Conv1d(128, 256, 3)
        self.conv_3_2 = nn.Conv1d(256, 256, 3)
        self.conv_3_3 = nn.Conv1d(512, 256, 3)
        self.conv_3_4 = nn.Conv1d(256, 256, 3)
        self.upconv_3 = nn.ConvTranspose1d(256, 128, 2)

        #FOURTH U-LAYER
        self.conv_4_1 = nn.Conv1d(256, 512, 3)
        self.conv_4_2 = nn.Conv1d(512, 512, 3)
        self.conv_4_3 = nn.Conv1d(1024, 512, 3)
        self.conv_4_4 = nn.Conv1d(512, 512, 3)
        self.upconv_4 = nn.ConvTranspose1d(512, 256, 2)

        #BOTTOM U-LAYER
        self.conv_5_1 = nn.Conv1d(512, 1024, 3)
        self.conv_5_2 = nn.Conv1d(1024, 1024, 3)
        self.upconv_5 = nn.ConvTranspose1d(1024, 512, 2)

        #BETWEEN LAYERS
        self.max_pool = nn.MaxPool1d(2)

        #ACTIVATION
        self.activation = nn.ReLU()

    def forward(self, x, t):

        #Positional encoding
        pos_enc = self.PE(t)

        # Down Layer 1
        x = self.activation(self.norm(self.conv_1_1(x), pos_enc))
        x = self.activation(self.norm(self.conv_1_2(x), pos_enc))
        x = self.max_pool(x)
        out1 = x.copy()

        # Down layer 2
        x = self.activation(self.norm(self.conv_2_1(x), pos_enc))
        x = self.activation(self.norm(self.conv_2_2(x), pos_enc))
        x = self.max_pool(x) 
        out2 = x.copy()

        # Down layer 3
        x = self.activation(self.norm(self.conv_3_1(x), pos_enc))
        x = self.activation(self.norm(self.conv_3_2(x), pos_enc))
        x = self.max_pool(x) 
        out3 = x.copy()

        # Down layer 4
        x = self.activation(self.norm(self.conv_4_1(x), pos_enc))
        x = self.activation(self.norm(self.conv_4_2(x), pos_enc))
        x = self.max_pool(x)
        out4 = x.copy()  

        # Bottom layer 
        x = self.activation(self.norm(self.conv_5_1(x), pos_enc))
        x = self.activation(self.norm(self.conv_5_2(x), pos_enc))
        x = self.upconv_5(x)

        # Up layer 4
        x = torch.cat((out4, x), dim = 1)
        x = self.activation(self.norm(self.conv_4_3(x), pos_enc))
        x = self.activation(self.norm(self.conv_4_4(x), pos_enc))
        x = self.upconv_4

        # Up layer 3
        x = torch.cat((out3, x), dim = 1)
        x = self.activation(self.norm(self.conv_3_3(x), pos_enc))
        x = self.activation(self.norm(self.conv_3_4(x), pos_enc))
        x = self.upconv_4

        # Up layer 2
        x = torch.cat((out2, x), dim = 1)
        x = self.activation(self.norm(self.conv_2_3(x), pos_enc))
        x = self.activation(self.norm(self.conv_2_4(x), pos_enc))
        x = self.upconv_4

        # Up layer 1
        x = torch.cat((out1, x), dim = 1)
        x = self.activation(self.norm(self.conv_1_3(x), pos_enc))
        x = self.activation(self.norm(self.conv_1_4(x), pos_enc))
        x = self.conv1x1(x)

        return x