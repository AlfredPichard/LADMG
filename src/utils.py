import torch
import os
from datetime import datetime
import json
from diffusion import UNetDiffusion

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
    LOG_DIR = "../log"

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

        self.log_path = os.path.join(self.LOG_DIR, self.name)

        self.last_checkpoint = last_checkpoint

    # Renvoie le dernier checkpoint pour la config donnée
    def get_last_checkpoint(self):
        last = None
        files = [f for f in os.listdir(self.SAVE_MODEL_DIR) if os.path.isfile(os.path.join(self.SAVE_MODEL_DIR, f))]
        for f in files:
            if f[-3:] == ".pt" and f.split("_iter")[0] == self.config.name:
                iteration = int(f.split("_iter")[0])
                if last is None or iteration > last:
                    last = iteration
        return last
    
    def get_model(self):
        if self.config.name == "diff_unet_simple_encodec_24":
            return UNetDiffusion(**self.config.args)
        else:
            raise Exception('CheckPointManager', f'{self.config.name} is an unvalid model configuration')           
    
    def get_loss(self):
        if self.config.loss == "MSE":
            return torch.nn.MSE()
        else: 
            raise Exception('CheckPointManager', f'{self.config.loss} is an unvalid loss')
    
#############################################
"""
Manager for training configuration, retrieves training configuration given a name for quicker initialisation
"""
#############################################
class TrainingManager:

    CONFIG_FILE = "../config/train.json"

    def __init__(self, name = "default"):
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

    def get_optimizer(self, model):
        if self.config.optimizer == "Adam":
            return torch.optim.Adam(**self.config.args)
        else: 
            raise Exception('TrainingManager', f'{self.config.optimizer} is an unvalid optimizer')


