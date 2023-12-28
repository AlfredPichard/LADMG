import encodec
import torch.nn as nn
import torch
import torch.nn.functional as F

class WrappedEncodec(nn.Module):

    def __init__(self,params_file=None) -> None:
        super().__init__()
        self.model = encodec.EncodecModel.encodec_model_24khz()
        self.sr = 24000
        self.param = False
        if params_file is not None:
            self.param = True
            pca_params = torch.load(params_file)
            self.latent_mean = pca_params["mean"]
            self.components = pca_params["components"]
            self.n_latents = 64
        
    def truncate(self, z: torch.Tensor):
        z = z-self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z,self.components.unsqueeze(-1).to(z))
        z = z[:,:self.n_latents]
        return z

    def encode(self, x: torch.Tensor, trunc = True) -> torch.Tensor:
        z,scales = self.model._encode_frame(x)
        z = z.transpose(0, 1)  # (n_quant, batch, time)
        z = self.model.quantizer.decode(z)
        if trunc == True and self.param:
            z = self.truncate(z)
        
        return z
               
    def decode(self, z: torch.Tensor,trunc=True) -> torch.Tensor: 
        if trunc==True and self.param:
            noise = torch.zeros(z.shape[0], 128-self.n_latents, z.shape[-1]).to(z)
            z = torch.cat((z, noise),axis=1)
            z = F.conv1d(z,self.components.T.unsqueeze(-1).to(z)) + self.latent_mean.unsqueeze(-1).to(z)
            
        return self.model.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
    

if __name__ == "__main__":
    encodec = WrappedEncodec()

    x = torch.randn(1,1,24000)

    z = encodec.encode(x)[0][0]
    x_rec = encodec.decode(z) #x_rec = encodec(x)

    print(z.shape,x_rec.shape)