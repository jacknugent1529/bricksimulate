from typing import Any
import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor
from einops import rearrange, einsum
from ..sequential_dataset import Node, Edge


class VoxelNet(nn.Module):
    def __init__(self, out_dim, activation = F.silu):
        super().__init__()
        
        self.activation = activation

        self.convs = nn.ModuleList([
            nn.Conv3d(1, 16, kernel_size=3, stride=2),
            nn.Conv3d(16, 64, kernel_size=3, stride=2),
            nn.Conv3d(64, 256, kernel_size=3, stride=2),
        ])

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x: Tensor):
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
        x = rearrange(x, "b c x y z -> b (c x y z)")
        
        # MLP
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

class GraphNet(nn.Module):
    def __init__(self, dim):
        super().__init__()

        def get_mlp():
            return nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
                nn.SiLU(),
            )

        self.convs = nn.ModuleList([
            gnn.GINEConv(get_mlp()),
            gnn.GINEConv(get_mlp()),
            gnn.GINEConv(get_mlp()),
        ])

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return x
    
class GraphEmbed(nn.Module):
    def __init__(self, dim, brick_embed):
        super().__init__()
        self.dim = dim

        self.pos_proj = nn.Sequential(
            nn.Linear(6, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

        # shift values range from -3 to 3
        self.edge_emb = nn.Embedding(7 * 7, dim)

        self.brick_emb = brick_embed

    def forward(self, graph):
        # contains x, pos, edge_attr
        # x - describes the brick
        # pos - position (use MLP)
        # edge_attr - x_shift/y_shift - use learned embedding

        x = self.brick_emb(graph.x) + self.pos_proj(graph.pos)

        edge_idx = (graph.edge_attr[:,0] + 3) * 7 + (graph.edge_attr[:,1] + 3)
        edge_attr = self.edge_emb(edge_idx.to(int))

        return x, edge_attr


class BrickEmbed(nn.Module):
    def __init__(self, num_bricks, dim):
        super().__init__()
        self.emb = nn.Embedding(num_bricks, dim)
    
    def to_idx(brick):
        return brick[:,1] // 90

    def forward(self, x):
        return self.emb(BrickEmbed.to_idx(x))
    
class EdgeIdx(nn.Module):
    def __init__(self, max_x_shift, max_y_shift, dim):
        super().__init__()
        self.num_x_shifts = (2 * max_x_shift + 1)
        self.max_x_shift = max_x_shift
        self.max_y_shift = max_y_shift
        self.num_embs = self.num_x_shifts * (2 * max_y_shift + 1) * 2

        self.emb = nn.Embedding(self.num_embs, dim)
    
    def to_idx(self, edge):
        outward = edge[:,0]
        x_shift = edge[:,1]
        y_shift = edge[:,2]
        return 2 * (
            (x_shift + self.max_x_shift) * self.num_x_shifts + 
            (y_shift + self.max_y_shift)
        ) + outward
    
    def forward(self, x):
        return self.emb(self.to_idx(x))
    
    
def add_voxel_repr(node_repr, voxels_repr):
    raise NotImplementedError()
    
class LegoNet(pl.LightningModule):
    """
    LegoNet basic model
    1. Obtain voxels representation using VoxelNet
    2. Obtain node-level representation using GraphNet
    3. Using the representation from the true node, and the voxels, predict logits for brick type
    4. Use voxels for graph and node representation to obtain logits for each node (representing the connection to be added)
        - this is the first loss
        - this is the second loss
    5. Using the true brick type, node, and voxel representations, predict the direction/x_shift/y_shift
    """
    def __init__(self, dim, num_bricks, l = 1, g = 1):
        super().__init__()

        self.save_hyperparameters()

        self.l = l
        self.g = g

        # represent the brick as a high-dimensional vector
        self.brick_repr = BrickEmbed(num_bricks, dim)

        # purpose is to transform graph x, pos, and edge_attr into vectors of consistent dimension
        self.graph_embed = GraphEmbed(dim, self.brick_repr)

        # purpose is to process the graph input, and obtain vector representations of each node
        self.node_repr = GraphNet(dim)

        # purpose is to process the voxels and obtain a single vector representation
        self.voxel_repr = VoxelNet(dim)

        # combine voxels and node representation
        self.voxel_node_repr = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )

        
        self.brick_choice = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        ) 
        self.brick_choice_linear = nn.Linear(dim, num_bricks)
        self.edge_node_choice_feat = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )

        self.edge_embed = EdgeIdx(3, 3, dim)
        self.edge_ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, self.edge_embed.num_embs)
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LegoNet")
        parser.add_argument("--dim", type=int, default=128)
        parser.add_argument("--l", type=int, default=1)
        parser.add_argument("--g", type=int, default=1)
        return parent_parser

    def training_step(self, batch):
        #TODO: think about when to move data to gpu
        voxels_repr = self.voxel_repr(batch.complete_voxels.unsqueeze(1))

        x, edge_attr = self.graph_embed(batch)
        edge_index = batch.edge_index
        graph_node_repr = self.node_repr(x=x, edge_index=edge_index, edge_attr=edge_attr)


        # concatenate appropriate voxels for model onto node representation
        voxel_model = torch.stack([voxels_repr[i] for i in batch.batch])
        voxel_node_combined = torch.cat((graph_node_repr, voxel_model), dim=1)
        voxels_node_repr = self.voxel_node_repr(voxel_node_combined)

        # select brick (node)
        brick_choice_repr = self.brick_choice(voxels_node_repr)
        graph_feat = gnn.global_mean_pool(brick_choice_repr, batch.batch) #NOTE: I don't think this is a great choice here; replace with RNN
        brick_logits = self.brick_choice_linear(graph_feat)

        true_brick = batch.y[:,:2]
        brick_loss = F.cross_entropy(brick_logits, BrickEmbed.to_idx(true_brick))



        # select node to be connected to
        true_brick_node = true_brick[batch.batch]
        brick_emb = self.brick_repr(true_brick_node) # using true brick
        edge_node_feats = self.edge_node_choice_feat(voxels_node_repr)
        sim = einsum(edge_node_feats, brick_emb, 'n f,n f -> n')

        edge_node_loss = 0
        true_node_idxs = batch.y[:,2]

        for i in range(len(batch.y)):
            edge_node_logits = sim[batch.batch == i]
            edge_node_loss += F.cross_entropy(edge_node_logits, true_node_idxs[i])

        # edge_node_loss = F.cross_entropy(edge_node_logits, batch.y[:,2])

        # select attributes of edge
        true_node_repr = voxels_node_repr[true_node_idxs]
        edge_attr_logits = self.edge_ff(true_node_repr)
        true_edge_idx = self.edge_embed.to_idx(batch.y[:,3:])
        edge_attr_loss = F.cross_entropy(edge_attr_logits, true_edge_idx)
    
        loss = brick_loss + self.l * edge_node_loss + self.g * edge_attr_loss

        self.log_dict({
            'train/brick_loss': brick_loss,
            'train/edge_node_loss': edge_node_loss,
            'train/edge_attr_loss': edge_attr_loss,
        })

        self.log_dict({'train/loss': loss,}, prog_bar=True)

        return loss
    
    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters())
