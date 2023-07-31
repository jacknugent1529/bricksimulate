from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from .dataset import LegoModel, prism_center
from heapq import heappop, heappush
from jaxtyping import Float, Int
from torch import Tensor
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from math import floor, ceil
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import io
from typing import List, NamedTuple, Tuple, Union
from torch_geometric.data import InMemoryDataset
from torch.utils.data import random_split
import torch
import os
import pickle
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader


class Node(NamedTuple):
    rot: float

class Edge(NamedTuple):
    node: int
    outward: bool
    x_shift: int
    y_shift: int

class SeqDataModule(pl.LightningDataModule):
    def __init__(self, datafolder, B, num_workers):
        super().__init__()

        self.save_hyperparameters()

        self.datafolder = datafolder
        self.B = B
        self.num_workers = num_workers
    
    def prepare_data(self) -> None:
        self.data = SequentialLegoData(self.datafolder)

        self.train_ds, self.val_ds = random_split(self.data, [0.8,0.2])
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, self.B, shuffle=True, num_workers=self.num_workers)
    
    # def val_dataloader(self):
    #     return DataLoader(self.train_ds, self.B, shuffle=True, num_workers=self.num_workers)


def get_brick_priorities(model: LegoModel) -> list[tuple[int,int,int,int]]:
    """
    Get brick priorities to build from the bottom of the model to the top
    - z,y,x, id

    """
    centers: Float[Tensor, "b x y z"] = prism_center(model.pos)
    def discretize(arr, step_size):
        bins = np.arange(floor(arr.min()), ceil(arr.max()), step_size)
        return np.digitize(arr, bins)
    x_disc = discretize(centers[:,0].numpy(), 2)
    y_disc = discretize(centers[:,1].numpy(), 2)
    z_disc = discretize(centers[:,2].numpy(), 1)

    priorities = {}
    min_priority = None
    for i, (x, y, z) in enumerate(zip(x_disc, y_disc, z_disc)):
        priorities[i] = (z,x,y,i)
        if min_priority is None or priorities[i] < min_priority:
            min_priority = priorities[i]
    
    return priorities, min_priority

def prioritized_search(g: Data, root: tuple[int,int,int], priorities: dict[int, tuple[int,int,int,int]], return_pred = False) -> list[int]:
    g_nx = nx.Graph(g.edge_index.T.to(int).tolist()).to_undirected()

    nodes_one_neighbor = 0
    for node in g_nx.nodes():
        if len(list(g_nx.neighbors(node))) == 1:
            nodes_one_neighbor += 1
    
    order = []
    q = [root]
    visited = set()
    pred = {}

    while len(q) > 0:
        _,_,_,curr = heappop(q)
        if curr in visited:
            continue
        order.append(curr)
        visited.add(curr)

        for n in g_nx.neighbors(curr):
            if n not in visited:
                heappush(q, priorities[n])
                pred[n] = curr
    
    assert order[0] == root[3]
    return order, pred

def restrict_model(obj: dict, nodes: list[int]) -> tuple[dict, dict]:
    # preserve the order within node_labels

    node_labels = []
    relabel = {}
    for i, node in enumerate(sorted(nodes)):
        relabel[node] = i
        node_labels.append(obj['node_labels'][node])

    edges = {(relabel[s],relabel[t]): v for ((s,t),v) in obj['edges'].items() if s in nodes and t in nodes}

    return {
        "node_labels": node_labels,
        "edges": edges
    }, relabel

def lego_obj_to_seq(obj: dict) -> list[tuple[LegoModel, tuple[Node, Edge]]]:
    model = LegoModel.from_obj(obj)
    priorities, root = get_brick_priorities(model)

    model_seq = []

    order, predecessors = prioritized_search(model, root, priorities)

    restricted_obj, relabel = restrict_model(obj, order[:1])
    curr_model = LegoModel.from_obj(restricted_obj)
    curr_nodes = [order[0]]
    for node in order[1:]:
        curr_nodes.append(node)
        pred = predecessors[node]
        if (node, pred) in obj['edges']:
            obj_edge = obj['edges'][(node, pred)]
            x_shift, y_shift = obj_edge['x_shift'], obj_edge['y_shift']
            edge = Edge(relabel[pred], False, x_shift, y_shift)
        elif (pred, node) in obj['edges']:
            obj_edge = obj['edges'][(pred, node)]
            x_shift, y_shift = obj_edge['x_shift'], obj_edge['y_shift']
            edge = Edge(relabel[pred], True, x_shift, y_shift)
        else:
            raise ValueError("Invalid transition")
        
        if obj['node_labels'][node] == 'Brick(2, 4)':
            node = Node(0)
        elif obj['node_labels'][node] == 'Brick(4, 2)':
            node = Node(90)
        else:
            raise ValueError("Invalid Brick")

        model_seq.append((curr_model, (node, edge)))

        restricted_obj, relabel = restrict_model(obj, curr_nodes)
        curr_model = LegoModel.from_obj(restricted_obj)
        
    return model_seq

def create_build_gif(steps: list[LegoModel], filename: str):
    # Set up the figure and axis
    ax = plt.figure().add_subplot(projection='3d')

    # Generate multiple images
    images = []
    duration = 100  # Duration of each frame in milliseconds

    for step, _ in tqdm(steps):
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

def lego_obj_to_build_gif(obj: dict, filename: str) -> list[LegoModel]:
    seq = [model for model, _ in lego_obj_to_seq(obj)]
    seq.append(LegoModel.from_obj(obj))
    create_build_gif(seq, filename)

def encode_brick(node: Node) -> int:
    if np.isclose(node.rot, 90):
        return 1
    else:
        return 0

def node_index(node: Node) -> int:
    return 1 if np.isclose(node.rot, 90) else 0

def edge_attr_index(edge: Edge) -> int:
    assert -3 <= edge.x_shift <= 3
    assert -3 <= edge.y_shift <= 3
    return 2 * ((3 + edge.x_shift) * 7 + edge.y_shift) + edge.outward
NUM_EDGE_ATTRS = 2 * 7 * 7

def encode_y(node: Node, edge: Edge) -> Int[Tensor, "y"]:
    y = torch.zeros(6, dtype=int)

    # brick embed - this should match dataset.py:from_obj
    y[0] = 0 # change when more bricks are added
    y[1] = node.rot
    

    # edge embed
    y[2] = edge.node
    y[3] = int(edge.outward)
    y[4] = edge.x_shift
    y[5] = edge.y_shift

    return y.reshape(1,-1)

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

    batch.edge_index
    - graph connectivity

    batch.y: B x 6
    - brick id
    - brick rotation
    - node connection id
    - edge outward (1 or 0)
    - edge x_shift
    - edge y_shift

    batch.batch: N
    - index of graph node belongs to (between 0 and B)
    
    """
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ["seq_torch_geo_data.pt"]
    
    def process(self):
        data_list = []

        path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(path, "rb") as f:
            data = pickle.load(f)

        model_valid = 0
        model_invalid = 0
        step_invalid = 0
        step_valid = 0

        print("Processing Data...")
        for obj in tqdm(data):
            try:
                model = LegoModel.from_obj(obj)
                complete_voxels = LegoModel.to_voxels(model, 1)
                if (np.array(complete_voxels.shape) > MAX_VOXEL).any():
                    # TODO: figure out better way to handle this
                    raise ValueError("Model too large")
                else:
                    complete_voxels = pad_voxels_to_shape(complete_voxels, MAX_VOXEL, MAX_VOXEL, MAX_VOXEL)
                
                model_valid += 1

                steps = lego_obj_to_seq(obj)
                for lego_obj, (node, edge) in steps:
                    y: Int[Tensor, "y"] = encode_y(node, edge)
                    lego_obj.y = y
                    # lego_obj.brick = Node(torch.Tensor(node.rot))
                    # lego_obj.edge = Edge(torch.Tensor([edge.node]), torch.Tensor([edge.outward]), torch.Tensor([edge.x_shift]), torch.Tensor([edge.y_shift]))
                    lego_obj.complete_voxels = complete_voxels.unsqueeze(0)

                    if LegoModel.model_valid(lego_obj):
                        data_list.append(lego_obj)
                        step_valid += 1
                    else:
                        step_invalid += 1
            except ValueError:
                model_invalid += 1
                continue
        
        data, slices = self.collate(data_list)

        print(f"Data Processed: {model_valid} models valid, {model_invalid} models invalid, {step_valid} steps valid, {step_invalid} steps invalid")

        torch.save((data, slices), self.processed_paths[0])

