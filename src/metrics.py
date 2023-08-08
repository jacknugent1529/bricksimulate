from torch_geometric.data import Data
import torch
from .utils import Prism
from .lego_model import LegoModel
from scipy.optimize import linear_sum_assignment

def match_boundaries(g1: Data, g2: Data):
    dx = g1.pos[:,0].min() - g2.pos[:,0].min()
    g2.pos[:,0] += dx
    g2.pos[:,3] += dx

    dy = g1.pos[:,1].min() - g2.pos[:,1].min()
    g2.pos[:,1] += dy
    g2.pos[:,4] += dy

    dz = g1.pos[:,2].min() - g2.pos[:,2].min()
    g2.pos[:,2] += dz
    g2.pos[:,5] += dz

def prism_IoU(p1: Prism, p2: Prism):
    corner1 = torch.stack([p1[:3], p2[:3]]).max(dim=0).values
    corner2 = torch.stack([p1[3:], p2[3:]]).min(dim=0).values

    vol = lambda p: torch.prod((p[3:] - p[:3]).clamp(min=0))
    int_vol = torch.prod((corner2 - corner1).clamp(min=0))
    assert int_vol >= 0, (corner1, corner2)
    assert vol(p1) > 0
    assert vol(p2) > 0
    union_vol = vol(p1) + vol(p2) - int_vol

    return int_vol / union_vol

def prisms_match_IoU(g1: Data, g2: Data):
    """
    Computes the average IoU by matching bricks in g1 to bricks in g2. This 
    prioritizes "alignment" in brick building patterns rather than the overall 
    volume. Matching is done with a variant of the Hungarian algorithm
    
    Both g1 and g2
    must include the 'pos' attribute describing their prisms
    """
    match_boundaries(g1, g2)
    ious = torch.zeros(len(g1.pos), len(g2.pos))
    
    for i, p1 in enumerate(g1.pos):
        for j, p2 in enumerate(g2.pos):
            ious[i,j] = prism_IoU(p1, p2)
    row_ind, col_ind = linear_sum_assignment(ious, maximize=True)

    cost = ious[row_ind, col_ind].sum()

    assert len(row_ind) == len(col_ind)
    return cost / max(len(g1.pos), len(g2.pos))
    
def voxelsIoU(g1: Data, g2: Data, resolution=1., *resolutions):
    match_boundaries(g1, g2)
    v1 = LegoModel.to_voxels(g1, resolution, *resolutions).to(torch.int)
    v2 = LegoModel.to_voxels(g2, resolution, *resolutions).to(torch.int)

    int = v1 & v2
    union = v1 | v2
    return int.sum() / union.sum()
