import torch
import os
from datetime import datetime

def save_checkpoint(model, model_name, dir_path = "../saved_models", with_date=True):
    if with_date:
        supp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    else:
        supp = ""

    model_path = model_name + "_" + supp + ".pt"
    model_path = os.path.join(dir_path, model_path)
    torch.save(obj=model.state_dict(),
             f=model_path)

    return model_path

def load_checkpoint(model, model_path):
    model.load_state_dict(torch.load(model_path))
