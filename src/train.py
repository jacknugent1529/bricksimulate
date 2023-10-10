from .sequential_dataset import JointGraphDataModule, MyToUndirected, LegoToUndirected
from .models.basic_model import LegoNet as BasicLegoNet
from .models.joint_graph_model import LegoNet as JointGraphLegoNet
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
import lightning.pytorch as pl
from argparse import ArgumentParser

trainer_callbacks = [
    ModelSummary(max_depth=2),
    ModelCheckpoint(
        save_top_k=2,
        monitor='val/loss',
        mode='min',
        save_last=True,
    ),
]

def run(args):
    #TODO: replace 200 with parameter
    data = JointGraphDataModule(args.data_folder, 1_000, args.batch_size, args.num_workers, transform=LegoToUndirected('mean'), include_gen_step=True, share_data=args.share_data)
    num_bricks = 3 # 2 rotations + STOP brick
    model = JointGraphLegoNet(args.dim, num_bricks, args.num_layers, args.l, args.g)

    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        fast_dev_run=args.fast_dev_run, 
        # precision=16,
        callbacks=trainer_callbacks, 
    )

    trainer.fit(model=model, datamodule=data)

def main():
    parser = ArgumentParser()
    
    # data
    parser.add_argument("--data-folder", type=str, default="data")
    parser.add_argument("-B","--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--randomize-order", action='store_true')
    parser.add_argument("--repeat", type=int)

    # training
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--share-data", action="store_true")
    parser.add_argument("--num-layers", type=int, default=3)

    parser = JointGraphLegoNet.add_model_specific_args(parser)

    args = parser.parse_args()

    run(args)
