import torch
import os
from datetime import datetime
import json

def save_checkpoint(model, model_name, dir_path = "../saved_models"):

    model_path = model_name + ".pt"
    model_path = os.path.join(dir_path, model_path)
    torch.save(obj=model.state_dict(),
             f=model_path)
    
def load_checkpoint(model, model_path):
    model.load_state_dict(torch.load(model_path))

#############################################
"""
Manager for model initialisation, retrieves model configuration given a name for quicker initialisation
"""
#############################################
class CheckPointManager:

    CONFIG_FILE = "../config/models.json"
    SAVE_MODEL_DIR = "../saved_models"

    def __init__(self, name, last_checkpoint = False):
        '''
        args:
            name: name of configuration in config/config.json
            last_checkpoint: if True retrieves last checkpoint for given configuration if exists
        '''
        f = open(self.CONFIG_FILE)
        data = json.load(f)
        
        self.config = None
        for jconfig in data:
            # On cherche une configuration ayant le nom "name"
            if jconfig.name == name:
                self.config = jconfig
        if self.config == None:
            try:
                # possibilité de faire passer l'indice dans config.json à la place du nom
                self.config = data[int(name)]
            except:
                # Si aucune configuration correspond, on prend la première
                self.config = data[0]
        f.close()

        last = self.get_last_checkpoint()
        if last:
            now = int(last) + 1
        else:
            now = 1

        self.name = self.config.name + "_iter" + now

        self.last_checkpoint = last_checkpoint

    # Renvoie le dernier checkpoint pour la config donnée
    def get_last_checkpoint(self):
        last = None
        files = [f for f in os.listdir(self.SAVE_MODEL_DIR) if os.path.isfile(os.path.join(self.SAVE_MODEL_DIR, f))]
        for f in files:
            if f[-3:] == ".pt" and f.split("_iter")[0] == self.config.name:
                iteration = int(f[-3:].split("_iter")[0])
                if last is None or iteration > last:
                    last = iteration
        return last
