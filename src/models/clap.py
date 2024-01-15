import torch
import numpy as np
import laion_clap

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

audio_data = None

audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)