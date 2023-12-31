import torch
import dataset as ds
import utils, train, diffusion

if __name__ == "__main__":

    DATA_PATH = "/data/atiam/triana/data/"

    parser = utils.Parser()
    cm = utils.CheckPointManager(name = parser.args.model, last_checkpoint=parser.args.checkpoint)
    tm = utils.TrainingManager(name = parser.args.train)
    model = cm.get_model()
    loss = cm.get_loss()
    tm = utils.TrainingManager()
    optimizer = tm.get_optimizer(model)
    trainer = train.Trainer(model, cm, optimizer, loss, None, None)
    
    dataset = ds.SimpleDataset(path=DATA_PATH, keys = ["encodec","metadata","clap"])

    batch = parser.args.batch

    train_dataset, valid_dataset= torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch, shuffle = True, collate_fn = ds.collate_fn_padd)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch, shuffle = True, collate_fn = ds.collate_fn_padd)

    tm = utils.TrainingManager()
    optimizer = tm.get_optimizer(model)

    trainer = train.Trainer(model, cm, optimizer, loss, None, None)

    if cm.last_checkpoint:
        trainer.load(cm.last_file)

    for i in range(10):
        d = next(iter(train_dataloader))
        z = d["encodec"]
        print(z.shape)
        t = torch.rand((batch))
        z_pred = model.forward(z, t)
        print(z_pred.shape)