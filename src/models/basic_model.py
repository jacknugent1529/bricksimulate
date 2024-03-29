from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Float, Int
from ..data_types import brick, buildstep, Brick, Edge, BuildStep, edge
import torchmetrics
from .generative_model import AbstractGenerativeModel

class VoxelNet(nn.Module):
    def __init__(self, out_dim, activation = F.silu):
        super().__init__()
        
        self.activation = activation

        self.convs = nn.ModuleList([
            nn.Conv3d(1, 16, kernel_size=3, stride=2),
            nn.Conv3d(16, 64, kernel_size=3, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, stride=2),
        ])

        self.fc1 = nn.Linear(1024, 512)
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

        GNNConv = gnn.GENConv
        self.convs = nn.ModuleList([
            GNNConv(dim, dim),
            GNNConv(dim, dim),
            GNNConv(dim, dim),
            # gnn.GINEConv(get_mlp()),
            # gnn.GINEConv(get_mlp()),
        ])

    def forward(self, x: Float[Tensor, "N dim"], edge_index: Int[Tensor, "2 M"], edge_attr: Float[Tensor, "M dim"]):
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return x
    
class GraphEmbed(nn.Module):
    def __init__(self, dim, brick_embed, max_x_shift, max_y_shift):
        super().__init__()
        self.dim = dim

        self.pos_proj = nn.Sequential(
            nn.Linear(6, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
            nn.BatchNorm1d(dim)
        )

        # shift values range from -3 to 3

        self.brick_emb = brick_embed

        self.max_x_shift = max_x_shift
        self.max_y_shift = max_y_shift
    
        self.edge_emb = nn.Embedding(self.num_edge_embeddings_undirected, dim)

    @property
    def num_edge_embeddings(self):
        return 2 * self.num_edge_embeddings_undirected
    
    @property
    def num_edge_embeddings_undirected(self):
        return (2 * self.max_x_shift + 1) * (2 * self.max_y_shift + 1)
    
    def to_edge_ids(self, edge_attr: Edge, top=None):
        x_shift = edge.x_shift(edge_attr)
        y_shift = edge.y_shift(edge_attr)
        num_x_shifts = 2 * self.max_x_shift + 1
        idx = (
            (x_shift + self.max_x_shift) * num_x_shifts + 
            (y_shift + self.max_y_shift)
        )
        if top is not None:
            idx = 2 * idx + top
        return idx.to(int)
    
    def from_edge_ids(self, edge_id, include_top=True):
        assert include_top, "not implemented"
        top = edge_id % 2
        edge_id = edge_id // 2
        num_x_shifts = 2 * self.max_x_shift + 1
        x_shift = (edge_id // num_x_shifts) - self.max_x_shift
        y_shift = (edge_id % num_x_shifts) - self.max_y_shift

        return top, x_shift, y_shift
    
    def embed_edges(self, edge_attr, top=None):
        return torch.Tensor(self.edge_emb(self.to_edge_ids(edge_attr, top=top)))

    def forward(self, graph):
        x = self.brick_emb(graph.x) + self.pos_proj(graph.pos)

        edge_attr = self.embed_edges(graph.edge_attr)

        return x, edge_attr


class BrickEmbed(nn.Module):
    def __init__(self, num_bricks, dim):
        super().__init__()
        self.num_bricks = num_bricks
        self.emb = nn.Embedding(num_bricks, dim)

    def to_idx(self, x):
        standard_idx = (1 + brick.rot(x) // 90).to(int)
        return torch.where(brick.id(x) == -1, 0, standard_idx)
    
    def from_idx(self, x):
        rot = (x - 1) * 90
        brick_id = torch.where(x == 0, -1, 0)
        rot = torch.where(x == 0, 0, (x - 1) * 90)
        return torch.cat((brick_id.reshape(-1,1), rot.reshape(-1,1)), dim=1)

    def embed_idx(self, x):
        return self.emb(x)

    def forward(self, x):
        return self.emb(self.to_idx(x))

class BrickChoiceAgent(nn.Module):
    def __init__(self, dim, num_bricks):
        super().__init__()
    
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )

        #TODO: could replace this with RNN
        self.graph_agg = lambda node_repr, batch: gnn.global_mean_pool(node_repr, batch)

        self.proj = nn.Linear(dim, num_bricks)
    
    def forward(self, node_repr: Float[Tensor, "N d"], batch: Int[Tensor, "N"]) -> Int[Tensor, "B brick"]:
        node_repr: Float[Tensor, "N d"] = self.ffn(node_repr)
        graph_repr: Float[Tensor, "b d"] = self.graph_agg(node_repr, batch)

        logits: Float[Tensor, "b num_bricks"] = self.proj(graph_repr) 

        return logits
    
class NodeChoiceAgent(nn.Module):
    """inspired by: https://docs.dgl.ai/en/0.8.x/tutorials/models/3_generative_model/5_dgmg.html#action-3-choose-a-destination"""
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )

        self.proj = nn.Linear(2 * dim, 1)
    
    def forward(self, 
        node_repr: Float[Tensor, "N d"], 
        brick_emb: Float[Tensor, "d"],
    ) -> Float[Tensor, "N"]:
        node_repr: Float[Tensor, "N d"] = self.ffn(node_repr)

        #TODO: Experiment with order
        node_brick_repr: Float[Tensor, "N d"] = torch.cat([node_repr, brick_emb], dim=1)

        logits: Float[Tensor, "N 1"] = self.proj(node_brick_repr)

        return logits

class EdgeChoiceAgent(nn.Module):
    def __init__(self, dim, num_edge_attrs):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
            nn.SiLU(),
            
        )

        self.proj = nn.Linear(dim, num_edge_attrs)

    def forward(self, node_repr, brick_repr):
        x = torch.cat([node_repr, brick_repr], dim=1)
        x = self.ffn(x)

        logits = self.proj(x)

        return logits

    
class LegoNet(pl.LightningModule, AbstractGenerativeModel):
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
        self.brick_embed = BrickEmbed(num_bricks, dim)

        # purpose is to transform graph x, pos, and edge_attr into vectors of consistent dimension
        self.graph_embed = GraphEmbed(dim, self.brick_embed, max_x_shift=3, max_y_shift=3)

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

        self.brick_choice_agent = BrickChoiceAgent(dim, num_bricks)
        self.node_choice_agent = NodeChoiceAgent(dim)
        self.edge_choice_agent = EdgeChoiceAgent(dim, self.graph_embed.num_edge_embeddings)

        self.val_metrics = {
            'brick': torchmetrics.Accuracy('multiclass', num_classes=num_bricks, average='macro'),
            'edge_node': torchmetrics.Accuracy('multiclass', average='micro', num_classes=1_000),
            'edge_attr': torchmetrics.Accuracy('multiclass', average='macro', num_classes=self.graph_embed.num_edge_embeddings)
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LegoNet")
        parser.add_argument("--dim", type=int, default=128, help='dimension of hidden vectors')
        parser.add_argument("--l", type=int, default=1, help='hyperparameter multiplier for edge_node_loss')
        parser.add_argument("--g", type=int, default=1, help='hyperparameter multiplier for edge_attr_loss')
        return parent_parser
    
    def process_graph(self, batch):
        voxels_repr: Tensor["b dim"] = self.voxel_repr(batch.complete_voxels.unsqueeze(1))

        x, edge_attr = self.graph_embed(batch)
        edge_index = batch.edge_index
        graph_node_repr: Tensor["N dim"] = self.node_repr(x=x, edge_index=edge_index, edge_attr=edge_attr)


        # concatenate appropriate voxels for model onto node representation
        voxel_model = torch.stack([voxels_repr[i] for i in batch.batch])
        voxel_node_combined = torch.cat((graph_node_repr, voxel_model), dim=1)
        voxels_node_repr = self.voxel_node_repr(voxel_node_combined)

        return voxels_node_repr


    def step(self, batch, prefix='train', metrics=None):
        voxels_node_repr: Float[Tensor, "N dim"] = self.process_graph(batch)

        # select brick (node)
        brick_logits = self.brick_choice_agent(voxels_node_repr, batch.batch)

        true_brick = buildstep.brick(batch.y)
        true_brick_idx = self.brick_embed.to_idx(true_brick).to(int)
        brick_loss = F.cross_entropy(brick_logits, true_brick_idx)
        if metrics is not None:
            metrics['brick'].update(brick_logits, true_brick_idx)

        # select node to be connected to
        true_brick_node = true_brick[batch.batch]
        true_brick_emb = self.brick_embed(true_brick_node.to(int)) # using true brick
        edge_node_logits = self.node_choice_agent(voxels_node_repr, true_brick_emb)

        edge_node_loss = 0
        true_node_idxs = buildstep.node_idx(batch.y)

        B = batch.num_graphs
        for i in range(B):
            graph_edge_node_logits = edge_node_logits[batch.batch == i].reshape(-1)
            edge_node_loss += F.cross_entropy(graph_edge_node_logits, true_node_idxs[i])

            if metrics is not None:
                metrics['edge_node'].update(graph_edge_node_logits.unsqueeze(0).argmax(dim=-1), true_node_idxs[i].unsqueeze(0))

        edge_node_loss /= B

        # select attributes of edge
        true_node_repr = torch.stack([voxels_node_repr[batch.batch == i][true_node_idxs[i]] for i in range(B)])
        edge_attr_logits = self.edge_choice_agent(true_node_repr, self.brick_embed(true_brick.to(int)))
        
        true_edge_idx = self.graph_embed.to_edge_ids(buildstep.edge(batch.y), top = buildstep.top(batch.y))
        edge_attr_loss = F.cross_entropy(edge_attr_logits, true_edge_idx)
        if metrics is not None:
            metrics['edge_attr'].update(edge_attr_logits, true_edge_idx)
    
        loss = brick_loss + self.l * edge_node_loss + self.g * edge_attr_loss

        self.log_dict({
            f'{prefix}/brick_loss': brick_loss,
            f'{prefix}/edge_node_loss': edge_node_loss,
            f'{prefix}/edge_attr_loss': edge_attr_loss,
        }, batch_size=B, add_dataloader_idx=False)

        self.log_dict({f'{prefix}/loss': loss,}, prog_bar=prefix == 'val', batch_size=B, add_dataloader_idx=False)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        if dataloader_idx == 0:
            B = batch.num_graphs
            self.step(batch, 'val', self.val_metrics)

            for k,v in self.val_metrics.items():
                self.log(f'val/{k}_acc', v.compute(), batch_size=B, add_dataloader_idx=False)
        else:
            # test full generation
            self.gen_step(batch, batch_idx, prefix='gen')
    
    def gen_step(self, batch, batch_idx, prefix):
        pred_step = self.gen_brick(batch)
        step = batch.y[0]
        
        brick_correct = (buildstep.brick(pred_step) == buildstep.brick(step)).all()
        node_correct = (buildstep.node_idx(pred_step) == buildstep.node_idx(step)).item()
        edge_correct = (buildstep.edge(pred_step) == buildstep.edge(step)).all() and (buildstep.top(pred_step) == buildstep.top(step)).item()

        self.log_dict({
            f'{prefix}/brick_acc': float(brick_correct), 
            f'{prefix}/edge_node_acc': float(node_correct),
            f'{prefix}/edge_attr_acc': float(edge_correct),
        }, batch_size=1, on_epoch=True, add_dataloader_idx=False, on_step=False)

        self.log_dict({
            f'{prefix}/acc': float(brick_correct and node_correct and edge_correct),
        }, batch_size=1, on_epoch=True, add_dataloader_idx=False, on_step=False, prog_bar=True)
        
    
    #TODO: improve
    # - randomly sample from distributions instead of argmax
    # - add a temperature parameter
    # - allow prediction on multiple graphs at once (primarily for testing purposes, not generation)
    def gen_brick(self, graph):
        assert graph.num_graphs == 1
        voxels_node_repr = self.process_graph(graph)

        # select brick (node)
        brick_logits = self.brick_choice_agent(voxels_node_repr, graph.batch)
        brick_idx = torch.argmax(brick_logits, dim=-1)

        # select node to be connected to
        true_brick_node = brick_idx[graph.batch]
        true_brick_emb = self.brick_embed.embed_idx(true_brick_node.to(int)) # using true brick
        edge_node_logits = self.node_choice_agent(voxels_node_repr, true_brick_emb)

        edge_node_idx = torch.argmax(edge_node_logits).unsqueeze(0)

        # select attributes of edge
        true_node_repr = voxels_node_repr[edge_node_idx]
        edge_attr_logits = self.edge_choice_agent(true_node_repr, self.brick_embed.embed_idx(brick_idx))

        edge_attr_idx = torch.argmax(edge_attr_logits)

        new_brick = self.brick_embed.from_idx(brick_idx)[0]
        top, x_shift, y_shift = self.graph_embed.from_edge_ids(edge_attr_idx, True)

        step = buildstep.new(new_brick, edge.new(x_shift, y_shift), edge_node_idx, top)
        return step

    
    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters())
