from torch import Tensor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import io
from torch_geometric.data import InMemoryDataset
from torch.utils.data import random_split
import torch
import os
import pickle
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.utils.undirected import to_undirected
from .lego_model import LegoModel, BuildStep
from .old_dataset import LegoModel as OldLegoModel

class MyToUndirected(BaseTransform):
    def __init__(self, reduce='mean'):
        self.reduce = reduce

    def __call__(self, data):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            store.edge_index, store.edge_attr = to_undirected(store.edge_index, store.edge_attr, reduce = self.reduce)
        
        return data

class SeqDataModule(pl.LightningDataModule):
    def __init__(self, datafolder, B, num_workers, transform=None):
        super().__init__()

        self.save_hyperparameters()

        self.datafolder = datafolder
        self.B = B
        self.num_workers = num_workers
        self.transform = transform
    
    def prepare_data(self) -> None:
        self.data = SequentialLegoData(self.datafolder, transform=self.transform)

        self.train_ds, self.val_ds = random_split(self.data, [0.8,0.2])
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, self.B, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.train_ds, self.B, shuffle=False, num_workers=self.num_workers)


def create_build_gif(steps: list[LegoModel], filename: str):
    # Set up the figure and axis
    ax = plt.figure().add_subplot(projection='3d')

    # Generate multiple images
    images = []
    duration = 100  # Duration of each frame in milliseconds

    for step in tqdm(steps):
        # Clear the axis and generate a new plot for each frame
        ax.clear()
        voxels = LegoModel.to_voxels(step, 1)
        ax.voxels(voxels, edgecolor='k')

        # Save the figure as an image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))

    # Save the images as a GIF with specified frame duration
    imageio.mimsave(filename, images, duration=duration)    


def pad_voxels_to_shape(X: Tensor, x, y, z) -> Tensor:
    rp = np.array([x, y, z]) - np.array(X.shape)

    return torch.nn.functional.pad(X, (0, rp[2], 0, rp[1], 0, rp[0]))

MAX_VOXEL = 25
class SequentialLegoData(InMemoryDataset):
    """
    When first run, this class will create a processed version of the dataset. If changes are made to the data and we want to re-run it, you can delete the `data/processed` folder.

    Data is in the following format:
    Define N to be the number of nodes, B to be the batch size (number of lego models), M to be the number of edges

    batch.x: N x 2 (node features)
    - brick id
    - brick rotation (0 or 90 degrees)

    batch.pos: N x 6 (node positions)
    - x,y,z,x,y,z - bounding box corners

    batch.edge_attr: M x 2 (edge features)
    - x_shift, y_shift

    batch.complete_voxels: B x 25 x 25 x 25

    batch.edge_index: 2 x M
    - graph connectivity

    batch.y: B x 6
    - brick id
    - brick rotation
    - node connection id
    - edge top (1 or 0)
    - edge x_shift
    - edge y_shift

    batch.batch: N
    - index of graph node belongs to (between 0 and B)
    
    """
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.transform = transform

    @property
    def raw_file_names(self):
        return ['dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ["seq_torch_geo_data.pt"]

    def get(self, idx):
        g = super().get(idx)

        if self.transform is not None:
            return self.transform(g)
        return g
    
    def process(self):
        data_list = []

        path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(path, "rb") as f:
            data = pickle.load(f)

        model_valid = 0
        model_invalid = 0
        old_model_invalid = 0
        step_invalid = 0
        step_valid = 0

        print("Processing Data...")
        for obj in tqdm(data):
            try:
                model = LegoModel.from_obj(obj)
                LegoModel.make_standard_order(model)
                complete_voxels = LegoModel.to_voxels(model)
                if (np.array(complete_voxels.shape) > MAX_VOXEL).any():
                    # TODO: figure out better way to handle this
                    raise ValueError("Model too large", complete_voxels.shape)
                else:
                    complete_voxels = pad_voxels_to_shape(complete_voxels, MAX_VOXEL, MAX_VOXEL, MAX_VOXEL)
                
                model_valid += 1

                steps = LegoModel.to_sequence(model)
                for lego_obj in steps:
                    lego_obj.complete_voxels = complete_voxels.unsqueeze(0)

                    if LegoModel.model_valid(lego_obj):
                        data_list.append(lego_obj)
                        step_valid += 1
                    else:
                        step_invalid += 1
            except ValueError as e:
                model_invalid += 1

                if e.args[0] == "Model too large":
                    old_model_invalid += 1
                else:
                    try: 
                        old_model = OldLegoModel.from_obj(obj)
                    except ValueError as e:
                        old_model_invalid += 1

                continue
        
        data, slices = self.collate(data_list)

        print(f"Data Processed: {model_valid} models valid, {model_invalid} models invalid, {step_valid} steps valid, {step_invalid} steps invalid")
        print(f"Old and new models invalid: {old_model_invalid}")

        torch.save((data, slices), self.processed_paths[0])

