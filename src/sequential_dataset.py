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
import numpy as np
import torch
import lightning.pytorch as pl
from jaxtyping import Float
from torch_geometric.loader import DataLoader
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils.undirected import to_undirected
import numpy as np
from torch_geometric.data import HeteroData, Data
from .lego_model import LegoModel, BuildStep
from .old_dataset import LegoModel as OldLegoModel
from .utils import PrismArr


class MyToUndirected(BaseTransform):
    def __init__(self, reduce='mean'):
        self.reduce = reduce

    def __call__(self, data):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            store.edge_index, store.edge_attr = to_undirected(store.edge_index, store.edge_attr, reduce = self.reduce)
        
        return data
    
class LegoToUndirected(BaseTransform):
    def __init__(self, reduce='mean'):
        self.reduce = reduce

    def __call__(self, data):
        store = data['lego', 'lego']
        if 'edge_index' not in store:
            return data

        store.edge_index, store.edge_attr = to_undirected(store.edge_index, store.edge_attr, reduce = self.reduce)
        
        return data

class SeqDataModule(pl.LightningDataModule):
    def __init__(self, datafolder, B, num_workers, include_gen_step=False, transform=None, share_data=False):
        super().__init__()

        self.save_hyperparameters()

        self.datafolder = datafolder
        self.B = B
        self.num_workers = num_workers
        self.transform = transform
        self.include_gen_step = include_gen_step
        self.shared_datasets = share_data
    
    def prepare_data(self) -> None:
        data = SequentialLegoData(self.datafolder, "train", transform=self.transform)
        data = SequentialLegoData(self.datafolder, "val", transform=self.transform)
        data = SequentialLegoData(self.datafolder, "gen", transform=self.transform)
    
    def train_dataloader(self):
        train_ds = SequentialLegoData(self.datafolder, "train", transform=self.transform)
        return DataLoader(train_ds, self.B, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        val_ds = SequentialLegoData(self.datafolder, "val", transform=self.transform)
        val = DataLoader(val_ds, self.B, shuffle=False, num_workers=self.num_workers)
        if self.include_gen_step:
            gen_ds = SequentialLegoData(self.datafolder, "gen", transform=self.transform)
            gen = DataLoader(gen_ds, 1, shuffle=False, num_workers = self.num_workers)
            return val, gen
        return val


class JointGraphDataModule(pl.LightningDataModule):
    def __init__(self, datafolder, n_points, B, num_workers, include_gen_step=False, transform=None, share_data=False, randomize_order=False, repeat=1):
        super().__init__()

        self.save_hyperparameters()

        self.datafolder = datafolder
        self.B = B
        self.num_workers = num_workers
        self.transform = transform
        self.include_gen_step = include_gen_step
        self.shared_datasets = share_data
        self.n_points = n_points
        self.random_order = randomize_order
        self.repeat = repeat
    
    def prepare_data(self) -> None:
        data = SequentialLegoDataJointGraph(self.datafolder, "train", self.n_points, transform=self.transform, random_order=self.random_order, repeat=self.repeat)
        data = SequentialLegoDataJointGraph(self.datafolder, "val", self.n_points, transform=self.transform, randomize_order=self.random_order, repeat=self.repeat)
        data = SequentialLegoDataJointGraph(self.datafolder, "gen", self.n_points, transform=self.transform, randomize_order=self.random_order, repeat=self.repeat)
    
    def train_dataloader(self):
        train_ds = SequentialLegoDataJointGraph(self.datafolder, "train", self.n_points, transform=self.transform, randomize_order=self.random_order, repeat=self.repeat)
        return DataLoader(train_ds, self.B, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        if self.shared_datasets:
            val_ds = SequentialLegoDataJointGraph(self.datafolder, "train", self.n_points, transform=self.transform, randomize_order=self.random_order, repeat=self.repeat)
        else:    
            val_ds = SequentialLegoDataJointGraph(self.datafolder, "val", self.n_points, transform=self.transform, randomize_order=self.random_order, repeat=self.repeat)
        val = DataLoader(val_ds, self.B, shuffle=False, num_workers=self.num_workers)
        if self.include_gen_step:
            if self.shared_datasets:
                gen_ds = SequentialLegoDataJointGraph(self.datafolder, "train", self.n_points, transform=self.transform, randomize_order=self.random_order, repeat=self.repeat)
            else:
                gen_ds = SequentialLegoDataJointGraph(self.datafolder, "gen", self.n_points, transform=self.transform, randomize_order=self.random_order, repeat=self.repeat)
            gen = DataLoader(gen_ds, 1, shuffle=False, num_workers = self.num_workers)
            return val, gen
        return val


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
    def __init__(self, root, split='train', transform=None):
        self.split = split
        super().__init__(root)
        match self.split:
            case 'train':
                processed_path = self.processed_paths[0]
            case 'val':
                processed_path = self.processed_paths[1]
            case 'gen':
                processed_path = self.processed_paths[2]
        self.data, self.slices, self.model_starts = torch.load(processed_path)
        self.transform = transform

    @property
    def raw_file_names(self):
        return ['dataset.pkl', 'splits.pkl']
    
    @property
    def processed_file_names(self):
        return ["seq_data_train.pt", "seq_data_val.pt", "seq_data_gen.pt"]
    
    def get(self, idx):
        g = super().get(idx)

        if self.transform is not None:
            return self.transform(g)
        return g
    
    def process(self):
        data_list = []
        model_starts = []

        with open(self.raw_paths[0], "rb") as f:
            data = pickle.load(f)
        
        with open(self.raw_paths[1], "rb") as f:
            split = pickle.load(f)
        match self.split:
            case 'train':
                data = [data[i] for i in split['train']]
                processed_path = self.processed_paths[0]
            case 'val':
                data = [data[i] for i in split['val']]
                processed_path = self.processed_paths[1]
            case 'gen':
                data = [data[i] for i in split['gen']]
                processed_path = self.processed_paths[2]

        model_valid = 0
        model_invalid = 0
        old_model_invalid = 0
        step_invalid = 0
        step_valid = 0

        print("Processing Data...")
        for obj in tqdm(data):
            model_starts.append(len(data_list))
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

        torch.save((data, slices, model_starts), processed_path)
    
def complete_bipartite_edges(m,n):
    row = torch.arange(m).reshape(-1,1).tile(n).reshape(-1)
    col = torch.arange(n).repeat(m)
    return torch.stack([row, col])

def make_joint_graph(lego: Data, point_cloud_graph: Data):
    """
    replace pos with point cloud graph
    """

    data = HeteroData()
    data['lego'].x = lego.x
    data['lego'].y = lego.y
    data['lego', 'lego'].edge_index = lego.edge_index
    data['lego', 'lego'].edge_attr = lego.edge_attr
    data['lego'].pos = lego.pos

    data['point'].pos = point_cloud_graph.pos
    data['point', 'point'].edge_index = point_cloud_graph.edge_index

    data['lego', 'point'].edge_index = complete_bipartite_edges(data['lego'].num_nodes, data['point'].num_nodes)
    data['point', 'lego'].edge_index = complete_bipartite_edges(data['point'].num_nodes, data['lego'].num_nodes)

    return data


def sample_point_cloud_from_prisms(prisms: PrismArr, N: int) -> Float[Tensor, "n 3"]:
    # TODO: generate points closer to surface
    prism_lens = torch.stack([(prisms[:,3] - prisms[:,0]), (prisms[:,4] - prisms[:,1]), (prisms[:,5] - prisms[:,2])], dim=1)
    volumes = prism_lens[:,0] * prism_lens[:,1] * prism_lens[:,2]
    ps = volumes / volumes.sum()

    prism_idxs = torch.from_numpy(np.random.choice(len(ps), N, p=ps.numpy()))
    placement_within_prism = torch.rand((N, 3))
    pos = prisms[prism_idxs,:3] + prism_lens[prism_idxs] * placement_within_prism
    return pos


class SequentialLegoDataJointGraph(InMemoryDataset):
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
    def __init__(self, root, split, n_points, transform=None, random_order=False, repeat=1):
        self.n_points = n_points
        self.split = split
        super().__init__(root)
        match self.split:
            case 'train':
                processed_path = self.processed_paths[0]
            case 'val':
                processed_path = self.processed_paths[1]
            case 'gen':
                processed_path = self.processed_paths[2]
        self.data, self.slices, = torch.load(processed_path)
        self.transform = transform
        self.random_order = random_order
        self.repeat = repeat

    @property
    def raw_file_names(self):
        return ['dataset.pkl', 'splits.pkl']
    
    @property
    def processed_file_names(self):
        if self.random_order:
            return ["seq_joint_graph_train.pt", "seq_joint_graph_val.pt", "seq_joint_graph_gen.pt"]
        else:
            return ["seq_joint_graph_train_random.pt", "seq_joint_graph_val_random.pt", "seq_joint_graph_gen_random.pt"]

    def get(self, idx):
        g = super().get(idx)

        if self.transform is not None:
            return self.transform(g)
        return g
    
    def process(self):
        data_list = []

        with open(self.raw_paths[0], "rb") as f:
            data = pickle.load(f)
        
        with open(self.raw_paths[1], "rb") as f:
            split = pickle.load(f)
        match self.split:
            case 'train':
                data = [data[i] for i in split['train']]
                processed_path = self.processed_paths[0]
            case 'val':
                data = [data[i] for i in split['val']]
                processed_path = self.processed_paths[1]
            case 'gen':
                data = [data[i] for i in split['gen']]
                processed_path = self.processed_paths[2]

        model_valid = 0
        model_invalid = 0
        old_model_invalid = 0
        step_invalid = 0
        step_valid = 0

        print("Processing Data...")
        for _ in range(self.repeat):
            for obj in tqdm(data[:5]):
                try:
                    model = LegoModel.from_obj(obj)
                    # TODO: see if this is necessary
                    LegoModel.make_standard_order(model)
                    point_cloud = sample_point_cloud_from_prisms(model.pos, self.n_points)
                    pg = Data(pos=point_cloud)
                    point_graph = KNNGraph()(pg)
                    
                    model_valid += 1

                    steps = LegoModel.to_sequence(model, self.random_order)
                    for lego_obj in steps:
                        step = make_joint_graph(lego_obj, point_graph)

                        if LegoModel.model_valid(lego_obj):
                            data_list.append(step)
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

        torch.save((data, slices), processed_path)
