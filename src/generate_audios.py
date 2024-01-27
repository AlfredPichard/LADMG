from diffusion import UNetDiffusion
import utils
import scipy.io.wavfile as wavefile
import numpy as np
import os
import io
import soundfile as sf

from models.unet_bt_conditioner import UNetBTConditioner

CHECKPOINT_PATH = ""

### Initalization
model = UNetDiffusion(
        n_layers=4,
        time_emb_dim=64,
        start_channels_base2=8,
        kernel_size_base2=1,
        n_groups=None,
        inference_bpm=125,
)
utils.load_checkpoint(model, CHECKPOINT_PATH)


def tensor_to_array(tensor, sample_rate):
    array = tensor.detach().numpy()
    array = array.squeeze()
    if abs(array).max() > 1:
        print("warning: audio amplitude out of range, auto clipped.")
        array = array.clip(-1, 1)
    assert array.ndim == 1, "input tensor should be 1 dimensional."
    array = (array * np.iinfo(np.int16).max).astype("<i2")

    import io
    import wave

    fio = io.BytesIO()
    with wave.open(fio, "wb") as wave_write:
        wave_write.setnchannels(1)
        wave_write.setsampwidth(2)
        wave_write.setframerate(sample_rate)
        wave_write.writeframes(array.data)
    audio_string = fio.getvalue()
    fio.close()

    data, samplerate = sf.read(io.BytesIO(audio_string))

    return data, samplerate

for i in range(10):
    # TODO: don't use a loop and generate with n_batch (bugs atm)
    inferences = model.inference()
    audio_tensor = inferences[0]
    audio, sr = tensor_to_array(audio_tensor, 24000)
    wavefile.write(
        f"generated_audio_{i}_125bpm.wav", 
        sr,
        audio)
    print(f"successfully generated audio {i}")