import torch
from .lego_model import LegoModel
from torch_geometric.data import InMemoryDataset
import os
import pickle

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
