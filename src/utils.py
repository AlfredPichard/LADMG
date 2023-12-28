import torch
import os
from datetime import datetime

def save_model(model, model_name, dir_path = "../saved_models", with_date=True):
    if with_date:
        supp = datetime.now().strftime("%S_%M_%H_%d_%m_*Y")
    else:
        supp = ""

    model_path = model_name + "_" + supp + ".pt"
    model_path = os.path.join(dir_path, model_path)
    torch.save(obj=model.state_dict(),
             f=model_path)

