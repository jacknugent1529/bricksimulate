import torch
from torch import Tensor
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset, HeteroData
from torch_geometric.transforms import KNNGraph, BaseTransform
from torch_geometric.utils import to_undirected
import pickle
from tqdm import tqdm
import numpy as np
from jaxtyping import Float
import random

from .lego_model import LegoModel
from .old_dataset import LegoModel as OldLegoModel
from .utils import PrismArr


class LegoToUndirected(BaseTransform):
    def __init__(self, reduce='mean'):
        self.reduce = reduce

    def __call__(self, data):
        store = data['lego', 'lego']
        if 'edge_index' not in store:
            return data

        store.edge_index, store.edge_attr = to_undirected(store.edge_index, store.edge_attr, reduce = self.reduce)
        
        return data

def complete_bipartite_edges(m,n,prob=0.1):
    row = torch.arange(m).reshape(-1,1).tile(n).reshape(-1)
    col = torch.arange(n).repeat(m)
    n = int(prob * len(row))
    indices = list(zip(row, col))
    random_pairs = random.sample(indices, n)
    random_pairs = torch.tensor(np.array(random_pairs)).T
    return random_pairs

def sample_point_cloud_from_prisms(prisms: PrismArr, N: int) -> Float[Tensor, "n 3"]:
    """Generate points uniformly within prisms"""
    prism_lens = torch.stack([(prisms[:,3] - prisms[:,0]), (prisms[:,4] - prisms[:,1]), (prisms[:,5] - prisms[:,2])], dim=1)
    volumes = prism_lens[:,0] * prism_lens[:,1] * prism_lens[:,2]
    ps = volumes / volumes.sum()

    prism_idxs = torch.from_numpy(np.random.choice(len(ps), N, p=ps.numpy()))
    placement_within_prism = torch.rand((N, 3))
    pos = prisms[prism_idxs,:3] + prism_lens[prism_idxs] * placement_within_prism
    return pos

def make_joint_graph(lego: Data, point_cloud_graph: Data) -> HeteroData:
    """
    Merge lego and point_cloud_graph
    """

    data = HeteroData()
    data['lego'].x = lego.x
    data['lego'].y = lego.y
    data['lego', 'lego'].edge_index = lego.edge_index
    data['lego', 'lego'].edge_attr = lego.edge_attr
    data['lego'].pos = lego.pos

    data['point'].pos = point_cloud_graph.pos
    data['point', 'point'].edge_index = point_cloud_graph.edge_index

    data['lego', 'point'].edge_index = complete_bipartite_edges(data['lego'].num_nodes, data['point'].num_nodes, prob=0.1)
    data['point', 'lego'].edge_index = complete_bipartite_edges(data['point'].num_nodes, data['lego'].num_nodes, prob=0.1)

    return data


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
        data = SequentialLegoDataJointGraph(self.datafolder, "val", self.n_points, transform=self.transform, random_order=self.random_order, repeat=self.repeat)
        data = SequentialLegoDataJointGraph(self.datafolder, "gen", self.n_points, transform=self.transform, random_order=self.random_order, repeat=self.repeat)
    
    def train_dataloader(self):
        train_ds = SequentialLegoDataJointGraph(self.datafolder, "train", self.n_points, transform=self.transform, random_order=self.random_order, repeat=self.repeat)
        return DataLoader(train_ds, self.B, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        if self.shared_datasets:
            val_ds = SequentialLegoDataJointGraph(self.datafolder, "train", self.n_points, transform=self.transform, random_order=self.random_order, repeat=self.repeat)
        else:    
            val_ds = SequentialLegoDataJointGraph(self.datafolder, "val", self.n_points, transform=self.transform, random_order=self.random_order, repeat=self.repeat)
        val = DataLoader(val_ds, self.B, shuffle=False, num_workers=self.num_workers)
        if self.include_gen_step:
            if self.shared_datasets:
                gen_ds = SequentialLegoDataJointGraph(self.datafolder, "train", self.n_points, transform=self.transform, random_order=self.random_order, repeat=self.repeat)
            else:
                gen_ds = SequentialLegoDataJointGraph(self.datafolder, "gen", self.n_points, transform=self.transform, random_order=self.random_order, repeat=self.repeat)
            gen = DataLoader(gen_ds, 1, shuffle=False, num_workers = self.num_workers)
            return val, gen
        return val


class SequentialLegoDataJointGraph(InMemoryDataset):
    """
    When first run, this class will create a processed version of the dataset. If changes are made to the data and we want to re-run it, you can delete the `data/processed` folder.

    Data is in the following format:
    Define N to be the number of nodes, B to be the batch size (number of lego models), M to be the number of edges

    `batch['lego'].x`: N x 2 (node features)
    - brick id
    - brick rotation (0 or 90 degrees)

    `batch['lego'].pos`: N x 6 (node positions)
    - x,y,z,x,y,z - bounding box corners

    `batch['lego'].edge_attr`: M x 2 (edge features)
    - x_shift, y_shift

    `batch['lego'].edge_index`: 2 x M
    - graph connectivity

    `batch['lego'].y`: B x 6
    - brick id
    - brick rotation
    - node connection id
    - edge top (1 or 0)
    - edge x_shift
    - edge y_shift

    `batch['point']`: KNN point cloud graph

    `batch[('point','to','lego')]`: edges from lego model to point cloud
    `batch[('lego','to','point')]`: edges from point cloud to lego model

    `batch.batch`: N
    - index of graph node belongs to (between 0 and B)
    
    """
    def __init__(self, root, split, n_points, transform=None, random_order=False, repeat=1):
        self.n_points = n_points
        self.split = split
        self.random_order = random_order
        self.repeat = repeat
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

        print(f"Processing Data... (repeat {self.repeat})")
        for _ in range(self.repeat):
            for obj in tqdm(data[:5]):
                try:
                    model = LegoModel.from_obj(obj)
                    # NOTE: this is necessary
                    LegoModel.make_standard_order(model)
                    point_cloud = sample_point_cloud_from_prisms(model.pos, self.n_points)
                    pg = Data(pos=point_cloud)
                    point_graph = KNNGraph(force_undirected=True)(pg)
                    
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
