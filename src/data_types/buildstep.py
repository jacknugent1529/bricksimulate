import torch
from torch import Tensor
from . import brick as brick_
from . import edge as edge_

BuildStep=Tensor

def new(brick: Tensor, edge: Tensor, node_idx: int, top: bool):
    return torch.cat([
        brick,
        edge,
        torch.Tensor([int(top), node_idx])
    ])

def brick(step) -> Tensor:
    return step[...,:brick_.len()]

def edge(step) -> Tensor:
    return step[...,brick_.len():brick_.len()+edge_.len()]

# @property
def node_idx(step):
    offset = brick_.len() + edge_.len()
    return step[...,offset + 1].to(int)

def to_str(step):
    return f"adding brick {brick(step)} to node {node_idx(step)}, top={top(step)}, edge_attr: {edge(step)}"

# @property
def top(step):
    offset = brick_.len() + edge_.len()
    return step[...,offset]

def get_stop_step() -> Tensor:
    return new(brick_.new(-1, 0), edge_.new(0,0), 0, False)
