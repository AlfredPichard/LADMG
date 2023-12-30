import torch 
import lmdb 
from audio_example import AudioExample
import numpy as np

class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, path,keys=['waveform','metadata'], transforms=None,readonly=True) -> None:
        super().__init__()
        self.env = lmdb.open(path, readonly=readonly)#, map_async=not True, writemap=not True)
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


        