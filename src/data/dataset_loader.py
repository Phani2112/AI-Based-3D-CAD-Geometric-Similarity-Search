import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from src.data.step_parser import STEPParser
from src.data.graph_builder import BRepGraphBuilder
import tqdm
import multiprocessing as mp

def _process_single_file(args):
    "Module-level function for multiprocessing pickling."
    step_dir, f = args
    try:
        path = os.path.join(step_dir, f)
        shape = STEPParser().parse(path)
        if shape is not None:
            graph = BRepGraphBuilder().build(shape)
            if graph is not None:
                return Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
    except Exception:
        pass
    return None

class FabWaveDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        
        processed_path = self.processed_paths[0]
        if os.path.exists(processed_path):
            self.data, self.slices = torch.load(processed_path, weights_only=False)
    
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
        
        step_files = []
        for category in os.listdir(self.root):
            step_dir = os.path.join(self.root, category, 'STEP', 'step final files')
            if not os.path.exists(step_dir):
                step_dir = os.path.join(self.root, category, 'STEP')
            if os.path.exists(step_dir):
                for f in os.listdir(step_dir):
                    if f.endswith('.stp') or f.endswith('.step'):
                        step_files.append((step_dir, f))
        
        print(f"Processing {len(step_files)} STEP files...")
        
        cpus = min(mp.cpu_count(), 8)
        print(f"Using {cpus} parallel workers")
        
        if cpus > 1:
            with mp.Pool(cpus) as pool:
                results = list(tqdm.tqdm(pool.imap(_process_single_file, step_files), total=len(step_files), desc="Generating graphs"))
                data_list = [r for r in results if r is not None]
        else:
            for step_dir, f in tqdm.tqdm(step_files, desc="Generating graphs"):
                result = _process_single_file((step_dir, f))
                if result is not None:
                    data_list.append(result)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved {len(data_list)} graphs")
