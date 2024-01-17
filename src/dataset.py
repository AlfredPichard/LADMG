import torch 
import lmdb 
from audio_example import AudioExample
import numpy as np
from scipy import signal


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, path,keys=['waveform','metadata'], transforms=None,readonly=True) -> None:
        super().__init__()
        self.env = lmdb.open(path, lock=False, readonly=readonly)#, map_async=not True, writemap=not True)
        with self.env.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
        self.buffer_keys = keys
    
    def collate_fn(self):
        def collate(L):
            out = {}
            for key in self.buffer_keys:
                if key != "metadata":
                    x = [a[key] for a in L]
                    x = np.stack(x)
                    x = torch.from_numpy(x).float()
                    out[key] = x
                else:
                    meta =  [a["metadata"] for a in L]
                    out["metadata"] = meta
                    
            return out
        
        return collate

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        ae = AudioExample()
        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index]))
            
        out = {}
        
        for key in self.buffer_keys:
            if key == "metadata":
                out[key] = ae.get_metadata()
            else:
                try:
                    out[key] = ae.get(key)
                except:
                    print("key: ",key," not found")
        return out


def to_pow2(x):
    return int(2**np.ceil(np.log2(x)))


def seq_len(seq):
    return seq.shape[-1]


def collate_fn_padd(batch, tensor_keys = ['encodec']):
    # handling batch creation in dataloader
    if len(batch) > 1:
        keys = batch[0].keys()
        out = {}
        for key in keys:
            out[key] = [item[key] for item in batch]

        # padding all latent variable sequences to closest power of 2 of longest sequence
        longest_seq_len = seq_len(max(out['encodec'], key=seq_len))
        out_seq_len = to_pow2(longest_seq_len)
        for n in range(len(batch)):
            x = out['encodec'][n]
            x_len = seq_len(x)
            to_pad = out_seq_len - x_len
            pad = (to_pad//2, to_pad - to_pad//2)
            out['encodec'][n] = torch.nn.functional.pad(torch.from_numpy(x), pad)
        for key in tensor_keys:
            out[key] = torch.stack(out[key])
        return out
    else:
        out = batch[0]
        x_len = seq_len(out['encodec'])
        out_seq_len = to_pow2(x_len)
        to_pad = out_seq_len - x_len
        pad = (to_pad//2, to_pad - to_pad//2)
        out['encodec'] = torch.nn.functional.pad(torch.tensor(out['encodec']), pad).unsqueeze(0)
        return out


def phasor(timesteps, encodec_sample_rate=24000, frame_sample_rate=320, n_frames=1024):
    resampled_timesteps = np.floor(np.array(timesteps)*encodec_sample_rate/frame_sample_rate).astype(int)
    sawtooth_like = np.zeros(n_frames)

    for k in range(1, len(resampled_timesteps)):
        try:
            t = np.linspace(0, 1, (resampled_timesteps[k] - resampled_timesteps[k-1]))
            sawtooth_like[resampled_timesteps[k-1]:resampled_timesteps[k]] = 0.5*signal.sawtooth(2 * np.pi * t)+0.5
        except ValueError:
            pass
    try:
        last_t = np.linspace(0, 1, max(0, min(resampled_timesteps[-1]-resampled_timesteps[-2], n_frames-resampled_timesteps[-1])))
        padding = np.zeros(max(0, n_frames-resampled_timesteps[-1]-len(last_t)))
        sawtooth_like[resampled_timesteps[-1]:n_frames] = np.concatenate((0.5*signal.sawtooth(2 * np.pi * last_t)+0.5, padding))
    except IndexError:
        pass
    
    return sawtooth_like


def phasor_from_bpm(bpm, encodec_sample_rate=24000, frame_sample_rate=320, n_frames=1024):
    total_length_seconds = n_frames*frame_sample_rate/encodec_sample_rate
    one_timestep = 60/bpm
    n_timesteps = np.floor(total_length_seconds/one_timestep).astype(int)
    timesteps = [one_timestep*i for i in range(n_timesteps)]
    return phasor(timesteps, encodec_sample_rate,frame_sample_rate, n_frames)
