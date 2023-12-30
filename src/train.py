import torch
from torch.utils.tensorboard import SummaryWriter
import utils

class Trainer:

    def __init__(self, model, manager, optimizer, loss_function, train_dataloader, valid_dataloader):
        self.model = model
        self.manager = manager
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.logger = SummaryWriter(self.manager.log_path)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.logger.add_graph(self.model)
        self.epoch = 0

    def train_one_epoch(self, log = False):
        running_loss = 0.0
        batch_size = self.train_dataloader.batch_size
        for i, data in enumerate(self.train_dataloader):
            z_1, meta = data
            a = torch.rand((batch_size))
            z_0 = self.model.sample(batch_size, z_1.shape[-1])
            z_a = ((1 - a)*z_0 + a*z_1)
            diff_pred = self.model(z_a, a)
            real_diff = z_1 - z_0
            loss = self.loss_function(diff_pred, real_diff)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        
        if log:
            self.logger.add_scalar('training loss',
                            running_loss / i,
                            self.epoch)
            running_loss = 0.0
        self.logger.flush()
        self.logger.close()
        self.epoch += 1

    def validate(self):
        running_loss = 0.0
        batch_size = self.valid_dataloader.batch_size
        for i, data in enumerate(self.valid_dataloader):
            z_1, meta = data
            a = torch.rand((batch_size))
            z_0 = self.model.sample(batch_size, z_1.shape[-1])
            z_a = ((1 - a)*z_0 + a*z_1)
            diff_pred = self.model(z_a, a)
            real_diff = z_1 - z_0
            loss = self.loss_function(diff_pred, real_diff)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        
        self.logger.add_scalar('validation loss',
                        running_loss / i,
                        self.epoch)
        running_loss = 0.0
        audio = self.model.inference()
        self.logger.add_audio('inference epoch ' + self.epoch, audio, sample_rate = self.model.encodec.sr)
        self.logger.flush()
        self.logger.close()

    def checkpoint(self):
        name = self.manager.name + "_epoch" + self.epoch
        utils.save_checkpoint(self.model, name)

    def load(self, model_path):
        utils.load_checkpoint(self.model, model_path)
