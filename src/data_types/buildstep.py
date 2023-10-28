"""
Functions to operate on build steps. Build steps represent how the next piece is added
Build steps are represented as a 1D tensor containing:
- brick: Brick - brick attributes (see brick.py)
- edge: Edge - edge attributes (see edge.py)
- node_idx: int - node the next piece is connected to
- top: bool - is the next piece placed above or below the current one

Intended usage in other modules:
```
import buildstep

b = buildstep.new(brick1,edge1,node1,top1)

buildstep.brick(b) # brick1
```
"""

import torch
from torch import Tensor
from . import brick as brick_
from . import edge as edge_

BuildStep=Tensor

def new(brick: Tensor, edge: Tensor, node_idx: int, top: bool) -> BuildStep:
    """Create new buildstep tensor"""
    return torch.cat([
        brick,
        edge,
        torch.Tensor([int(top), node_idx])
    ])

def brick(step: BuildStep) -> brick_.Brick:
    """retrieve brick from buildstep; can be broadcast over tensors"""
    return step[...,:brick_.len()]

def edge(step: BuildStep) -> edge_.Edge:
    """retrieve edge from buildstep; can be broadcast over tensors"""
    return step[...,brick_.len():brick_.len()+edge_.len()]

def node_idx(step: BuildStep) -> Tensor:
    """retrieve node_idx from buildstep; can be broadcast over tensors"""
    offset = brick_.len() + edge_.len()
    return step[...,offset + 1].to(int)

def top(step: BuildStep) -> Tensor:
    """retrieve `top` attribute from buildstep; can be broadcast over tensors"""
    offset = brick_.len() + edge_.len()
    return step[...,offset]

def to_str(step: BuildStep) -> str:
    """create string describing a single buildstep"""
    return f"adding brick {brick(step)} to node {node_idx(step)}, top={top(step)}, edge_attr: {edge(step)}"

def get_stop_step() -> BuildStep:
    """Buildstep representing the end of the build sequence. Denoted by brick id `-1`"""
    return new(brick_.new(-1, 0), edge_.new(0,0), 0, False)
