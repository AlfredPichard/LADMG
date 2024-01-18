import torch
from torch.utils.tensorboard import SummaryWriter
import utils
import numpy as np
import dataset as ds

class Trainer:

    DEVICE = 'cuda:2'

    def __init__(self, model, manager, optimizer, loss_function, train_dataloader, valid_dataloader):
        self.model = model
        self.manager = manager
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.logger = SummaryWriter(self.manager.log_path)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        """self.logger.add_graph(self.model)
        self.logger.flush()
        self.logger.close()"""
        self.epoch = 0

    def train_one_epoch(self, log = True):
        running_loss = 0.0
        batch_size = self.train_dataloader.batch_size
        for i, data in enumerate(self.train_dataloader):
            metadata = data['metadata']
            numpy_bt_conditioner = self.batch_dropout(
                batch_size=batch_size, 
                dropout_p=0.17,
                data=np.array([ds.phasor(metadata[k]['BT_beat']) for k in range(batch_size)]))
            bt_conditioner = torch.from_numpy(numpy_bt_conditioner)[:,None,:].float().to(self.DEVICE)         
            z_1 = data['encodec'].to(self.DEVICE)
            a = torch.rand((batch_size), device=self.DEVICE).view(batch_size, 1, 1)
            z_0 = self.model.sample(batch_size, z_1.shape[-1])
            z_a = ((1 - a)*z_0 + a*z_1)

            diff_pred = self.model(z_a, a, bt_conditioner)
            real_diff = z_1 - z_0
            loss = self.loss_function(diff_pred, real_diff)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            #if i > 10:
            #    break
        
        if log:
            print(f'Epoch {self.epoch + 1}, training loss: {running_loss / (i+1)}')
            self.logger.add_scalar('training loss',
                running_loss / (i+1),
                self.epoch)
            self.logger.flush()
            self.logger.close()
        self.epoch += 1

    def validate(self):
        running_loss = 0.0
        batch_size = self.valid_dataloader.batch_size
        for i, data in enumerate(self.valid_dataloader):
            metadata = data['metadata']
            bt_conditioner = torch.from_numpy(np.array([ds.phasor(metadata[k]['BT_beat']) for k in range(batch_size)]))[:,None,:].float().to(self.DEVICE)
            z_1 = data['encodec'].to(self.DEVICE)
            a = torch.rand((batch_size), device=self.DEVICE).view(batch_size, 1, 1)
            z_0 = self.model.sample(batch_size, z_1.shape[-1])
            z_a = ((1 - a)*z_0 + a*z_1)
            
            diff_pred = self.model(z_a, a, bt_conditioner)
            real_diff = z_1 - z_0
            loss = self.loss_function(diff_pred, real_diff)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            #if i > 10:
            #    break
        
        print(f'Epoch {self.epoch}, validation loss: {running_loss / (i+1)}')
        self.logger.add_scalar('validation loss',
                        running_loss / (i+1),
                        self.epoch)
        audio = self.model.inference()
        self.logger.add_audio("audio samples", audio, sample_rate = self.model.encodec.sr, global_step = self.epoch )
        self.logger.flush()
        self.logger.close()

    def checkpoint(self):
        name = self.manager.model_path + "_epoch" + str(self.epoch)
        #name = self.manager.name
        utils.save_checkpoint(self.model, name)

    def load(self, model_path):
        print(f'Loading from {model_path}')
        utils.load_checkpoint(self.model, model_path)

    def batch_dropout(self, batch_size, dropout_p, data):
        index = np.random.randint(batch_size)
        u = np.random.random()
        if u < dropout_p:
            data[index] = -1*np.ones(data[index].shape)

        return data


if __name__ == "__main__":

    import dataset as ds

    path = "/data/nils/minimal/encodec_24k_BT"
    dataset = ds.SimpleDataset(path=path, keys = ["encodec","metadata"])

    batch = 16

    train_dataset, valid_dataset= torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch, shuffle = True, collate_fn = ds.collate_fn_padd)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch, shuffle = True, collate_fn = ds.collate_fn_padd)

    cm = utils.CheckPointManager()
    model = cm.get_model()
        
    loss = cm.get_loss()

    tm = utils.TrainingManager()
    optimizer = tm.get_optimizer(model)

    trainer = Trainer(model, cm, optimizer, loss, None, None)

    data = next(iter(train_dataloader))
    z = data["encodec"]
    metadata = data['metadata']
    bt_conditioner = torch.from_numpy(np.array([ds.phasor(metadata[k]['BT_beat']) for k in range(batch_size)]))[:,None,:].float()
    print(z.shape)
    print(bt_conditioner.shape)
    t = torch.rand((batch))
    z_pred = model.forward(z, t, bt_conditioner)
    print(z_pred.shape)
