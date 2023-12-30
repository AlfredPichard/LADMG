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
            # Search configuration with name "name"
            if jconfig.name == name:
                self.config = jconfig
        if self.config == None:
            try:
                # Possbilility to pass index argument instead of name
                self.config = data[int(name)]
            except:
                # If no configuration corresponds, return first configuration (which is default configuration)
                self.config = data[0]
        f.close()

        last_file, last_iter = self.get_last_checkpoint()

        # now is the number of iterations of models saved for given configuration
        if last_iter:
            now = int(last_iter) + 1
        else:
            now = 1

        # last_file is the last saved model file for the given configuration, if it exists
        if last_file:
            self.last_file = os.path.join(self.SAVE_MODEL_DIR, last_file)
        else:
            self.last_file = None

        self.last_epoch = int(self.last_file[:-3].split("_epoch")[-1])

        self.name = self.config.name + "_iter" + now

        self.log_path = os.path.join(self.LOG_DIR, self.name)

        self.last_checkpoint = last_checkpoint

    # Renvoie le dernier checkpoint pour la config donnée
    def get_last_checkpoint(self):
        last = None, None
        files = [f for f in os.listdir(self.SAVE_MODEL_DIR) if os.path.isfile(os.path.join(self.SAVE_MODEL_DIR, f))]
        for f in files:
            if f[-3:] == ".pt" and f.split("_iter")[0] == self.config.name:
                iteration = int(f.split("_iter")[0])
                if last is None or iteration > last:
                    last = f, iteration
        return last
    
    def get_model(self):
        if self.config.model == "UNetDiffusion":
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

    def get_optimizer(self):
        if self.config.optimizer == "Adam":
            return torch.optim.Adam(**self.config.args)
        else: 
            raise Exception('TrainingManager', f'{self.config.optimizer} is an unvalid optimizer')


