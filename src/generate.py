from torch_geometric.data import Data
from .data_types import Brick, brick, Edge, edge, BuildStep, buildstep
import torch
from torch import Tensor
from .models.model_abc import AbstractGenerativeModel
from jaxtyping import Int
from .lego_model import LegoModel
from torch_geometric.data import Batch
from .sequential_dataset import pad_voxels_to_shape, MAX_VOXEL


#TODO: Improve. 
# 1. randomly sample from probability distributions instead of argmax
# 2. add temperature parameter
# 3. don't immediately terminate on invalid brick (try drawing random samples N times and use first valid brick placement)
def generate_single_model(
    model: AbstractGenerativeModel, 
    voxels: Int[Tensor, "x y z"],
    first_brick: Brick,
    keep_seq=False
) -> LegoModel:
    voxels = pad_voxels_to_shape(voxels, MAX_VOXEL, MAX_VOXEL, MAX_VOXEL)
    g = Data(
        x = first_brick.reshape(1,-1),
        pos = brick.get_prism(first_brick).reshape(1,-1),
        edge_attr=torch.zeros([0,edge.len()]),
        edge_index=torch.zeros([2,0]).to(int),
        complete_voxels=voxels.unsqueeze(0)
    )

    if keep_seq:
        seq = []

    while True:
        batch = Batch.from_data_list([g])
        step = model.gen_brick(batch)
        new_brick = buildstep.brick(step)
        if brick.id(new_brick) == -1:
            break

        edge_attr = buildstep.edge(step)
        node_idx = buildstep.node_idx(step)
        top = buildstep.top(step)

        if keep_seq:
            seq.append(batch.clone())
        try:
            LegoModel.add_piece(g, new_brick, edge_attr, node_idx.to(int), top)
        except:
            print("Invalid piece placement; terminate generation")
            break
    
    if keep_seq:
        return batch, seq
    return batch

