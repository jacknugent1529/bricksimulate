from torch_geometric.data import Data
from ..data_types import Brick, brick, Edge, edge, BuildStep, buildstep
import torch
from torch import Tensor
from jaxtyping import Int
from ..lego_model import LegoModel
from torch_geometric.data import Batch
from ..sequential_dataset import pad_voxels_to_shape, MAX_VOXEL


def complete_bipartite_edges(m,n):
    row = torch.arange(m).reshape(-1,1).tile(n).reshape(-1)
    col = torch.arange(n).repeat(m)
    return torch.stack([row, col])

class AbstractGenerativeModel():
    def gen_brick(self, graph) -> BuildStep:
        raise NotImplementedError()
    
    #TODO: IMPROVE
    # - add a temperature parameter
    # - don't immediately terminate on invalid brick (try drawing random samples N times and use first valid brick placement)
    def generate_single_lego_model(
        self,
        current_model: Data | None = None,
        first_brick: Brick | None = None,
        voxels: Int[Tensor, "x y z"] | None = None,
        keep_seq=False,
        transform=None,
        verbose=False,
    ) -> LegoModel:
        if current_model is None:
            voxels = pad_voxels_to_shape(voxels, MAX_VOXEL, MAX_VOXEL, MAX_VOXEL)
            current_model = Data(
                x = first_brick.reshape(1,-1),
                pos = brick.get_prism(first_brick).reshape(1,-1),
                edge_attr=torch.zeros([0,edge.len()]),
                edge_index=torch.zeros([2,0]).to(int),
                complete_voxels=voxels.unsqueeze(0)
            )

        if keep_seq:
            seq = []

        i = 0
        while i < 20:
            if transform is not None:
                current_model = transform(current_model)
            batch = Batch.from_data_list([current_model])
            step = self.gen_brick(batch)
            new_brick = buildstep.brick(step)
            if verbose:
                print(f"{i}: {buildstep.to_str(step)}")
            if brick.id(new_brick) == -1:
                break

            edge_attr = buildstep.edge(step)
            node_idx = buildstep.node_idx(step)
            top = buildstep.top(step)

            if keep_seq:
                seq.append(batch.clone())
            try:
                tmp_model = Data(
                    x=current_model['lego'].x,
                    edge_index=current_model['lego','lego'].edge_index,
                    edge_attr=current_model['lego','lego'].edge_attr,
                    pos=current_model['lego'].pos
                )
                LegoModel.add_piece(tmp_model, new_brick, edge_attr, node_idx.to(int), top)
                
                current_model['lego'].x = tmp_model.x
                current_model['lego'].pos = tmp_model.pos
                current_model['lego','lego'].edge_index = tmp_model.edge_index
                current_model['lego','lego'].edge_attr = tmp_model.edge_attr


                
                current_model['lego', 'point'].edge_index = complete_bipartite_edges(current_model['lego'].num_nodes, current_model['point'].num_nodes)
                current_model['point','lego'].edge_index = complete_bipartite_edges(current_model['point'].num_nodes, current_model['lego'].num_nodes)

                print(current_model)
            except:
                print("Invalid piece placement; terminate generation")
                break
            i += 1
        
        print(f"Generated {i} steps")
        
        if keep_seq:
            return batch, seq
        return batch

