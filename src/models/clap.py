import torch
import torch.nn as nn
import numpy as np
import laion_clap

class CLAP(nn.Module):

    def __init__(self, enable_fusion=False, checkpoint=None):
        super(CLAP, self).__init__()
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        if checkpoint:
            try:
                pass
            except Exception as e:
                print(e)
                self.model = None
        else:
            self.model.load_ckpt()

    def embedding_from_waveform(self, x):
        audio_data = torch.from_numpy(self.int16_to_float32(self.float32_to_int16(x))).float()
        audio_embed = self.model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
        return audio_embed
    
    def embedding_from_files(self, audio_files):
        audio_embed = self.model.get_audio_embedding_from_filelist(x = audio_files, use_tensor=True)
        return audio_embed
    
    def embedding_from_text(self, text):
        text_embed = self.model.get_text_embedding(text, use_tensor=True)
        return text_embed

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)


    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

if __name__ == '__main__':
    clap = CLAP()
    print('done')