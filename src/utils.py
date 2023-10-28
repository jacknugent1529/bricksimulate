"""
Utility functions for working with prism objects representing the lego bricks in 3D space

And other random stuff
"""

from torch_geometric.utils import to_scipy_sparse_matrix
import torch
from jaxtyping import Float, Bool
from torch import Tensor

# always have x1 < x2, y1 < y2, z1 < z2
PrismArr = Float[Tensor, "b xyz_xyz"]
Prism = Float[Tensor, "xyz_xyz"]
Point = Float[Tensor, "xyz"]

def check_collision_point(p: Point, prisms: PrismArr) -> Bool[Tensor, "b"]:
    """
    Check if point p collides with elements of `prisms`
    """
    x_collide = (prisms[...,0] <= p[0]) & (prisms[...,3] >= p[0])
    y_collide = (prisms[...,1] <= p[1]) & (prisms[...,4] >= p[1])
    z_collide = (prisms[...,2] <= p[2]) & (prisms[...,5] >= p[2])

    return x_collide & y_collide & z_collide

eps = 1e-6
def check_intersection(prism: Prism, other_prisms: PrismArr) -> Bool[Tensor, "b"]:
    """
    Check interesection between a prism and a tensor of other prisms
    """
    x_collide = (prism[0] + eps < other_prisms[:,3]) & (prism[3] - eps > other_prisms[:,0])
    y_collide = (prism[1] + eps < other_prisms[:,4]) & (prism[4] - eps > other_prisms[:,1])
    z_collide = (prism[2] + eps < other_prisms[:,5]) & (prism[5] - eps > other_prisms[:,2])

    return x_collide & y_collide & z_collide

def prism_center(prisms: PrismArr) -> Float[Tensor, "b xyz"]:
    """
    Get the center of prisms
    """
    x = (prisms[...,3] + prisms[...,0]) / 2
    y = (prisms[...,4] + prisms[...,1]) / 2
    z = (prisms[...,5] + prisms[...,2]) / 2

    return torch.stack((x,y,z)).T

def scale_prism(prism: Prism, x=1, y=1, z=1) -> Prism:
    """
    Scale a prism while keeping its center the same
    """
    center = prism_center(prism[None,:])
    c = torch.stack([center, center]).reshape(-1)

    prism_origin = prism - c
    prism_origin *= torch.Tensor([x, y, z, x, y, z])

    scaled = prism_origin + c
    return scaled

def check_intersection_small(prism: Prism, other_prisms: PrismArr) -> Bool[Tensor, "b"]:
    """
    Check intersection while scaling `prism` down slightly so that intersection exactly along an 
    edge won't be counted
    """
    small_prism = scale_prism(prism, 0.99, 0.99, 0.99)

    return check_intersection(small_prism, other_prisms)

# NOTE: unoptimized; taken from 
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/largest_connected_components.html
def graph_connected(data, connection='strong') -> bool:
    import scipy.sparse as sp
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

    num_components, component = sp.csgraph.connected_components(
        adj, connection=connection)

    return num_components == 1
