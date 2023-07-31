from .sequential_dataset import SeqDataModule
from .models.basic_model import LegoNet
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
import lightning.pytorch as pl
from argparse import ArgumentParser

trainer_callbacks = [
    ModelSummary(max_depth=2),
    ModelCheckpoint(
        save_top_k=2,
        monitor='val/acc',
        mode='max',
        save_last=True,
    ),
]

def run(args):
    data = SeqDataModule(args.data_folder, args.batch_size, args.num_workers)
    model = LegoNet(args.dim, 2, args.l, args.g)

    trainer = pl.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=1, fast_dev_run=args.fast_dev_run, callbacks=trainer_callbacks)

    trainer.fit(model=model, datamodule=data)

def main():
    parser = ArgumentParser()
    
    # data
    parser.add_argument("--data-folder", type=str)
    parser.add_argument("-B","--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=3)

    # training
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--fast-dev-run", action="store_true")


    parser = LegoNet.add_model_specific_args(parser)

    args = parser.parse_args()

    run(args)
