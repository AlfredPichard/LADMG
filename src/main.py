import torch
import dataset as ds
import utils, train, diffusion
import sys
from torchinfo import summary

if __name__ == "__main__":

    DATA_PATH = "/data/nils/minimal/encodec_24k"

    ### Initalization
    print("Initializing params...")
    parser = utils.Parser()
    cm = utils.CheckPointManager(name = parser.args.model, last_checkpoint=parser.args.checkpoint)
    tm = utils.TrainingManager(name = parser.args.train)
    model = cm.get_model()
    loss = cm.get_loss()
    tm = utils.TrainingManager()
    optimizer = tm.get_optimizer(model)
    trainer = train.Trainer(model, cm, optimizer, loss, None, None)
    
    dataset = ds.SimpleDataset(path=DATA_PATH, keys=['encodec'], transforms=None, readonly=True)

    batch = parser.args.batch

    train_dataset, valid_dataset= torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch, shuffle = True, collate_fn = ds.collate_fn_padd, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch, shuffle = True, collate_fn = ds.collate_fn_padd, drop_last=True)

    tm = utils.TrainingManager()
    optimizer = tm.get_optimizer(model)

    trainer = train.Trainer(model, cm, optimizer, loss, train_dataloader, valid_dataloader)

    ### Print model summary
    model_summary = summary(model, input_size=((batch, 128, 512) ,(batch,1,1)))
    print(f"Model Summary : {model_summary}")

    ### Load last model state
    if cm.last_checkpoint:
        trainer.load(cm.last_file)
        trainer.epoch = cm.last_epoch
        print(f"Loaded existing model {cm.last_file}")

    ### Train loop
    try:
        print("Training...")
        if parser.args.epochs:
            for i in range(parser.args.epochs):
                log = (trainer.epoch == parser.args.epochs_log)
                trainer.train_one_epoch(log = True)
                if log:
                    trainer.validate()
            print('Training stopping, saving model')
            trainer.checkpoint()
            sys.exit()
        else:
            while True:
                log = ((trainer.epoch + 1) % parser.args.epochs_log == 0 and trainer.epoch > 0)
                trainer.train_one_epoch(log = True)
                if log:
                    trainer.validate()
    except KeyboardInterrupt:
        print('Training stopping, saving model')
        trainer.checkpoint()
        sys.exit()