import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from src.data.step_parser import STEPParser
from src.data.graph_builder import BRepGraphBuilder

class FabWaveDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.parser = STEPParser()
        self.builder = BRepGraphBuilder()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['fabwave_data.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        for category in os.listdir(self.root):
            step_dir = os.path.join(self.root, category, 'STEP', 'step final files')
            if not os.path.exists(step_dir):
                step_dir = os.path.join(self.root, category, 'STEP')
            if os.path.exists(step_dir):
                for f in os.listdir(step_dir):
                    if f.endswith('.stp') or f.endswith('.step'):
                        shape = self.parser.parse(os.path.join(step_dir, f))
                        if shape is not None:
                            graph = self.builder.build(shape)
                            if graph is not None:
                                data = Data(
                                    x=graph.x,
                                    edge_index=graph.edge_index,
                                    edge_attr=graph.edge_attr
                                )
                                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])