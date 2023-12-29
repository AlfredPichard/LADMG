import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, model, optimizer, loss_function, train_dataloader, valid_dataloader, log_path = '../log'):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.logger = SummaryWriter(log_path)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.logger.add_graph(self.model)

    def train_one_epoch(self, epoch = 0, log = False, inference = False):
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
                            epoch * len(self.train_dataloader) + i)
            running_loss = 0.0
        if inference:
            audio = self.model.inference()
            self.logger.add_audio('inference epoch ' + epoch, audio, sample_rate = self.model.encodec.sr)
            self.logger.flush()
            self.logger.close()

