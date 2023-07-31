from torch.utils.data import Dataset
import pickle
from torch_geometric.data import Data
from torch import Tensor
from jaxtyping import Float, Bool
import torch
import networkx as nx
from math import ceil
from torch_geometric.data import InMemoryDataset
from typing import Union
import os
from tqdm import tqdm

# always have x1 < x2, y1 < y2, z1 < z2
PrismArr = Float[Tensor, "b xyz_xyz"]
Prism = Float[Tensor, "xyz_xyz"]
Point = Float[Tensor, "xyz"]

def check_collision_point(p: Point, prisms: PrismArr) -> Bool[Tensor, "b"]:
    x_collide = (prisms[:,0] <= p[0]) & (prisms[:,3] >= p[0])
    y_collide = (prisms[:,1] <= p[1]) & (prisms[:,4] >= p[1])
    z_collide = (prisms[:,2] <= p[2]) & (prisms[:,5] >= p[2])

    return x_collide & y_collide & z_collide

eps = 1e-6
def check_intersection(prism: Prism, other_prisms: PrismArr) -> Bool[Tensor, "b"]:
    x_collide = (prism[0] + eps < other_prisms[:,3]) & (prism[3] - eps > other_prisms[:,0])
    y_collide = (prism[1] + eps < other_prisms[:,4]) & (prism[4] - eps > other_prisms[:,1])
    z_collide = (prism[2] + eps < other_prisms[:,5]) & (prism[5] - eps > other_prisms[:,2])

    return x_collide & y_collide & z_collide

def prism_center(prisms: PrismArr) -> Float[Tensor, "b xyz"]:
    x = (prisms[:,3] + prisms[:,0]) / 2
    y = (prisms[:,4] + prisms[:,1]) / 2
    z = (prisms[:,5] + prisms[:,2]) / 2

    return torch.stack((x,y,z)).T

def dfs_bidirectional_preorder(g: Data, return_graph = False) -> list:
    g_nx = nx.DiGraph(g.edge_index.T.to(int).tolist())
    if len(g.edge_index) == 0:
        order = [i for i in range(len(g.x))]
    else:
        order = list(nx.dfs_preorder_nodes(g_nx.to_undirected()))
    if return_graph:
        return order, g_nx
    return order

class LegoBrick():
    def __init__(self, brick_id, rot):
        self.brick_id = brick_id
        self.rot = rot

        self.prism = self.get_prism()
        self.placed = False
        
    def get_prism(self) -> Prism:
        if self.brick_id != 0:
            raise ValueError("Only 2x4 bricks currently supported")
        if self.rot == 0:
            #TODO: figure out bug with changing z height
            return torch.Tensor([0,0,0,2,4,1])
        elif abs(self.rot) == 90:
            return torch.Tensor([0,0,0,4,2,1])
        else:
            raise ValueError("Only 90 degree rotations supported")
    
    def place_on(self, other: "LegoBrick", x_shift: int, y_shift: int, top: bool) -> bool:
        """Returns true if the placement is consistent (i.e. isn't different than a previous placement)"""
        if self.placed:
            prism_orig = self.prism.clone()
        
        if not top:
            x_shift *= -1
            y_shift *= -1

        dx, dy, dz = prism_center(other.prism[None,:])[0] - prism_center(self.prism[None,:])[0]
        self.prism[[0,3]] += dx + x_shift
        self.prism[[1,4]] += dy + y_shift
        self.prism[[2,5]] += dz
        
        # if intersect, move up
        if check_intersection(self.prism, other.prism.unsqueeze(0)).item():
            z_shift = other.prism[5] - other.prism[2]
            if not top:
                z_shift *= -1
            self.prism[[2,5]] += z_shift
        
        if self.placed and not torch.isclose(self.prism, prism_orig).all():
            # conflicting positions
            return False

        self.placed = True
        return True


class LegoModel(Data):
    """
    Lego Representation

    X (node features): 
    - brick id
    - brick orientation range (-90, 90] degrees
    
    edge_index (src,dst): dst is placed on top of src
    
    edge_attr (edge features):
    - x_shift
    - y_shift

    pos (bounding box):
    - x1,y1,z1,x2,y2,z2

    NOTE: currently only supports models with a single connected component
    TODO: add checks for this
    """
    def from_obj(obj: dict) -> "LegoModel":
        # convert to torch geometric
        X = []
        for node in obj['node_labels']:
            if node == 'Brick(2, 4)':
                X.append([0, 0])
            elif node == 'Brick(4, 2)':
                X.append([0, 90])
            else:
                assert False, "invalid brick"
        
        X = torch.Tensor(X).to(int)
        
        edge_index = []
        edge_attr = []
        for (src, dst), edge in obj['edges'].items():
            edge_index.append([src,dst])
            edge_attr.append([edge['x_shift'], edge['y_shift']])

        edge_index = torch.Tensor(edge_index).T
        edge_attr = torch.Tensor(edge_attr)

        model = Data(x = X, edge_index=edge_index.to(int), edge_attr=edge_attr)
        model.pos = LegoModel.get_prisms(model)

        return model
    
    def to_voxels(self, *resolution: Union[float, tuple[float,float,float]]):
        """
        Resolution represents the dimensions of an individual voxel. Can be specified by a single number, which is used for all dimensions, or 3 x/y/z resolutions. 

        Larger resolution is "coarser", smaller is ore fine
        """
        if len(resolution) == 3:
            res_x, res_y, res_z = resolution
        elif len(resolution) == 1:
            res_x = res_y = res_z = resolution[0]
        else:
            raise ValueError("Give 1 or 3 resolution arguments")

        min_x = self.pos[:,0].min() - res_x
        max_x = self.pos[:,3].max() + res_x
        min_y = self.pos[:,1].min() - res_y
        max_y = self.pos[:,4].max() + res_y
        min_z = self.pos[:,2].min() - res_z
        max_z = self.pos[:,5].max() + res_z

        boxes_x = ceil((max_x - min_x) / res_x)
        boxes_y = ceil((max_y - min_y) / res_y)
        boxes_z = ceil((max_z - min_z) / res_z)

        voxels = torch.zeros(boxes_x, boxes_y, boxes_z)

        for i,x in enumerate(torch.linspace(min_x, max_x, boxes_x)):
            for j,y in enumerate(torch.linspace(min_y, max_y, boxes_y)):
                for k,z in enumerate(torch.linspace(min_z, max_z, boxes_z)):
                    p = torch.Tensor([x,y,z])
                    voxels[i,j,k] = check_collision_point(p, self.pos).any()
        
        return voxels
        
    def get_prisms(self) -> PrismArr:
        if len(self.x) == 0:
            return torch.Tensor([])
        edge_features = {(int(k[0]), int(k[1])):v for k,v in zip(self.edge_index.T, self.edge_attr)}
        
        # topo sort
        order, g_nx = dfs_bidirectional_preorder(self, return_graph = True)
        first_brick = self.x[order[0]]
        bricks = {order[0]: LegoBrick(*first_brick)}
        for node in order[1:]:
            brick = LegoBrick(*self.x[node])
            for pred in g_nx.predecessors(node):
                if pred not in bricks:
                    continue
                x_shift, y_shift = edge_features[(pred, node)]
                other = bricks[pred]

                if not brick.place_on(other, x_shift, y_shift, top=True):
                    raise ValueError("Invalid Brick Placement")
            for succ in g_nx.successors(node):
                if succ not in bricks:
                    continue
                x_shift, y_shift = edge_features[(node, succ)]

                other = bricks[succ]

                if not brick.place_on(other, x_shift, y_shift, top=False):
                    raise ValueError("Invalid Brick Placement")

            bricks[node] = brick

        prism_list = []
        for i in range(len(self.x)):
            prism_list.append(bricks[i].prism)

        prisms = torch.stack(prism_list)

        return prisms
    
    def model_valid(self) -> bool:
        for i, prism in enumerate(self.pos):
            intersections = check_intersection(prism, self.pos)
            mask = torch.ones_like(intersections).to(bool)
            mask[i] = False

            if (intersections & mask).any():
                return False

        return True

# taken from here: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
class LegoData(InMemoryDataset):
    """
    When first run, this class will create a processed version of the dataset. If changes are made to the data and we want to re-run it, you can delete the `data/processed` folder.
    """
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['torch_geo_data.pt']
    
    def process(self):
        data_list = []

        path = os.path.join(self.raw_dir,self.raw_file_names[0])
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        for obj in data:
            try:
                model = LegoModel.from_obj(obj)
                if LegoModel.model_valid(model):
                    data_list.append(model)
            except ValueError:
                continue
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
