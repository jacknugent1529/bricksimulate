from torch_geometric.data import Data
from typing import NamedTuple, Union, Optional
from math import ceil, floor
import torch
from . import utils
from dataclasses import dataclass
from torch import Tensor
from heapq import heappop, heappush
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import numpy as np
from torch_geometric.utils.subgraph import subgraph
from .data_types import brick, edge, buildstep, Brick, Edge, BuildStep, BRICK_HEIGHT
 
    
def prioritized_search(g: Data, root_idx: int, priorities: dict[int,any]):
    g_dir: nx.Graph = to_networkx(g, node_attrs=['x'], edge_attrs=['edge_attr'])
    g_undir = g_dir.to_undirected()

    # priority, curr, pred
    q = [(priorities[root_idx], root_idx, -1)]
    visited_idx = set()

    while len(q) > 0:
        _, curr, pred = heappop(q)
        if curr in visited_idx:
            continue

        visited_idx.add(curr)
        
        for n in g_undir.neighbors(curr):
            if n not in visited_idx:
                t = (priorities[n], n, curr)
                heappush(q, t)

        if pred == -1:
            continue

        if (pred, curr) in g_dir.edges:
            top = True
            edge_attr = g_dir.edges[pred,curr]['edge_attr']
        else:
            top = False
            edge_attr = g_dir.edges[curr,pred]['edge_attr']
        yield curr, torch.Tensor(g_dir.nodes[curr]['x']), torch.Tensor(edge_attr), pred, top


def default_priorities(n):
    return {i:i for i in range(n)}

def random_priorities(n):
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    # TODO: random, but not as random as shuffle

    return {i:idxs[i] for i in range(n)}

def get_brick_priority_order(model: "LegoModel") -> dict[int, any]:
    centers = utils.prism_center(model.pos)

    def discretize(arr, step_size):
        bins = np.arange(floor(arr.min()), ceil(arr.max()), step_size)
        return np.digitize(arr, bins)
    x_disc = discretize(centers[:,0].numpy(), 2)
    y_disc = discretize(centers[:,1].numpy(), 2)
    z_disc = discretize(centers[:,2].numpy(), 1)

    dt = np.dtype([('x', np.int32), ('y', np.int32), ('z', np.int32)])

    priorities_list = np.array(list(zip(x_disc, y_disc, z_disc)), dtype=dt)
    
    return np.argsort(priorities_list, order=('z','y','x'))

class LegoModel(Data):
    """

    X (node features): 
    - brick id
    - brick orientation range (-90, 90] degrees
    
    edge_index (src,dst): dst is placed on top of src
    
    edge_attr (edge features):
    - x_shift
    - y_shift

    pos (bounding box):
    - x1,y1,z1,x2,y2,z2
    """
    def from_obj(obj: dict) -> "LegoModel":
        X = []
        for label in obj['node_labels']:
            X.append(brick.from_str(label))
        
        X = torch.stack(X).to(int)

        edge_index = []
        edge_attr = []
        for (src, dst), edge_dict in obj['edges'].items():
            edge_index.append([src,dst])
            edge_attr.append(edge.new(edge_dict['x_shift'], edge_dict['y_shift']))

        edge_index = torch.Tensor(edge_index).T
        edge_attr = torch.stack(edge_attr)

        model = Data(x = X, edge_index=edge_index.to(int), edge_attr=edge_attr)
        model.pos = LegoModel.build(model)

        return model

    def build(data):
        """
        Calculates `pos`
        """
        first_brick = data.x[0]
        pos = torch.zeros((len(data.x), 6))
        pos[0] = brick.get_prism(first_brick)
        
        priorities = default_priorities(len(data.x)) # avoid using 'standard' priorities unless necessary
        graph_traversal = prioritized_search(data, 0, priorities)
        for curr_node_idx, b, e, prev_node_idx, top in graph_traversal:
            assert curr_node_idx < len(data.x)
            pos[curr_node_idx] = LegoModel.place_piece(pos, b, e, prev_node_idx, top)
        
        return pos

    def make_standard_order(self):
        new_to_old = torch.from_numpy(get_brick_priority_order(self))

        old_to_new = torch.zeros_like(new_to_old)
        for new, old in enumerate(new_to_old):
            old_to_new[old] = new
        
        self.x = self.x[new_to_old]
        
        for i in range(self.edge_index.shape[1]):
            self.edge_index[0,i] = old_to_new[self.edge_index[0,i]]
            self.edge_index[1,i] = old_to_new[self.edge_index[1,i]]
        self.pos = LegoModel.build(self)

    
    def to_sequence(data, random_order=False):
        if random_order:
            priorities = random_priorities(len(data.x)) # avoid using 'standard' priorities unless necessary
        else:
            priorities = default_priorities(len(data.x)) # avoid using 'standard' priorities unless necessary
        graph_traversal = prioritized_search(data, 0, priorities)
        visited = torch.zeros(len(data.x)).to(bool)
        visited[0] = True
        for curr_node_idx, brick, edge, prev_node_idx, top in graph_traversal:
            edge_index, edge_attr = subgraph(visited, data.edge_index, data.edge_attr, relabel_nodes=False)

            node_idx_map = torch.zeros(visited.shape[0]).to(int)
            node_idx_map[visited] = torch.arange(visited.sum().item())

            edge_index = node_idx_map[edge_index]

            prev_node_idx = node_idx_map[prev_node_idx]

            step = Data(
                x = data.x[visited],
                pos = data.pos[visited],
                edge_index = edge_index,
                edge_attr=edge_attr,
            )

            step.y = buildstep.new(brick, edge, prev_node_idx, top).unsqueeze(0)
            
            yield step

            visited[curr_node_idx] = True

        # full graph
        final_step = data.clone()
        final_step.y = buildstep.get_stop_step().unsqueeze(0)

        yield final_step
        


    def place_piece(pos, b: Brick, e: Edge, other_idx: int, top: bool):
        other_prism = pos[other_idx]
        assert len(other_prism) == 6, other_prism.shape
        prism = brick.get_prism(b)
        
        # center prism to be same as other
        delta = (utils.prism_center(other_prism[None,:]) - utils.prism_center(prism[None,:]))[0]
        prism[:3] += delta
        prism[3:] += delta

        # add shifts
        c = 1 if top else -1
        prism[[0,3]] += edge.x_shift(e) * c
        prism[[1,4]] += edge.y_shift(e) * c

        assert utils.check_intersection(prism, other_prism[None,:]).item()
        if top:
            dz = other_prism[5] - prism[2]
        else:
            dz = other_prism[2] - prism[5]
        prism[[2,5]] += dz

        #TODO: Fix stack here
        if utils.check_intersection_small(prism, pos).any():
            raise ValueError("invalid brick placement")

        return prism

    def generate_connections(self, pos, new_node_idx):
        scaled = utils.scale_prism(pos, z = 1.1)
        intersections = utils.check_intersection(scaled, self.pos)

        for node_idx in torch.argwhere(intersections):
            if node_idx == new_node_idx:
                continue
            c1 = utils.prism_center(pos[None,:])[0]
            c2 = utils.prism_center(self.pos[node_idx])[0]
            x_shift, y_shift, z_shift = (c2 - c1)
            
            e = edge.new(x_shift, y_shift)
            if z_shift > 0:
                new_edge_index = torch.Tensor([node_idx, new_node_idx])[:,None]
            else:
                e *= -1
                new_edge_index = torch.Tensor([new_node_idx, node_idx])[:,None]

            self.edge_attr = torch.cat([self.edge_attr, e[None,:]])
            self.edge_index = torch.cat([self.edge_index, new_edge_index.to(int)], dim=1)

    def add_piece(data, b: Brick, e: Edge, other_idx: int, top: bool):
        # node attributes
        new_node_idx = len(data.x)
        data.x = torch.cat([data.x, b[None,:]])

        new_pos = LegoModel.place_piece(data.pos, b, e, other_idx, top)
        if utils.check_intersection_small(new_pos, data.pos).any():
            raise ValueError("Brick intersects with another")
        data.pos = torch.cat([data.pos, new_pos[None,:]])

        LegoModel.generate_connections(data, new_pos, new_node_idx)

    def model_valid(self) -> bool:
        for i, prism in enumerate(self.pos):
            intersections = utils.check_intersection_small(prism, self.pos)
            mask = torch.ones_like(intersections).to(bool)
            mask[i] = False

            if (intersections & mask).any():
                return False

        return True

    def to_voxels(self, *resolution: Union[float, tuple[float,float,float]], focus=Optional[Tensor]):
        """
        Resolution represents the dimensions of an individual voxel. Can be specified by a single number, which is used for all dimensions, or 3 x/y/z resolutions. 

        Larger resolution is "coarser", smaller is ore fine
        """
        if len(resolution) == 3:
            res_x, res_y, res_z = resolution
        elif len(resolution) == 1:
            res_x = res_y = res_z = resolution[0]
        elif len(resolution) == 0:
            res_x = res_y = 1
            res_z = BRICK_HEIGHT
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
        if focus is not None:
            focus_collision = torch.zeros_like(voxels)
            focus_arr = torch.zeros(len(self.pos)).to(int)
            focus_arr[focus] = 1


        for i,x in enumerate(torch.linspace(min_x, max_x, boxes_x)):
            for j,y in enumerate(torch.linspace(min_y, max_y, boxes_y)):
                for k,z in enumerate(torch.linspace(min_z, max_z, boxes_z)):
                    p = torch.Tensor([x,y,z])
                    collision = utils.check_collision_point(p, self.pos)
                    voxels[i,j,k] = collision.any()

                    if focus is not None:
                        focus_collision[i,j,k] = (collision & focus_arr).any()
        if focus is not None:
            return voxels, focus_collision
        return voxels