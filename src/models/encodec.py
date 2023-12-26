from datasets import load_dataset, Audio
from transformers import AutoProcessor, EncodecModel
import torch
import torch.nn as nn

class Encodec(nn.Module):

    def __init__(self, model_id = "facebook/encodec_24khz"):
        super(Encodec, self).__init__()
        self.model_id = model_id
        self.model = EncodecModel.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
    def test(self):
        print('test')
    
    def process(self, audio):
        return self.processor(raw_audio=audio, sampling_rate=self.processor.sampling_rate, return_tensors='pt')
    
    def encode(self, inputs):
        return self.model.encode(inputs["input_values"], inputs["padding_mask"])
    
    def decode(self, codes, inputs):
        return self.model.decode(codes.audio_codes, codes.audio_scales, inputs["padding_mask"])[0]
    
    def get_codes(self, inputs):
        encoded_frames = self.encode(inputs).audio_codes
        return torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

    def get_codes_audio(self, audio):
        inputs = self.process(audio)
        encoded_frames = self.encode(inputs).audio_codes
        return torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
    
if __name__ == "__main__":
    model = Encodec()
    # dummy dataset, however you can swap this with an dataset on the 🤗 hub or bring your own
    librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # cast the audio data to the correct sampling rate for the model
    librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=model.processor.sampling_rate))
    audio_sample = librispeech_dummy[10]["audio"]["array"]

    codes = model.get_codes_audio(audio_sample)
    print(codes.shape)

