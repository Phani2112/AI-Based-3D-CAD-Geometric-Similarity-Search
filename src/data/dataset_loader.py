import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from src.data.step_parser import STEPParser
from src.data.graph_builder import BRepGraphBuilder
import tqdm
import multiprocessing as mp

def _process_single_file(args):
    "Module-level function for multiprocessing pickling."
    if len(args) == 2:
        step_dir, f = args
        category = os.path.basename(os.path.dirname(step_dir))
    else:
        step_dir, f, category = args
    try:
        path = os.path.join(step_dir, f)
        shape = STEPParser().parse(path)
        if shape is not None:
            graph = BRepGraphBuilder().build(shape)
            if graph is not None:
                uuid = os.path.splitext(f)[0]
                return {
                    "data": Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr),
                    "metadata": {"uuid": uuid, "category": category, "filename": f, "path": path},
                }
    except Exception:
        pass
    return None


def _is_step_file(filename):
    return filename.lower().endswith(('.stp', '.step'))


def _discover_step_files(root):
    flat_step_dir = os.path.join(root, 'step')
    if os.path.isdir(flat_step_dir):
        category = os.path.basename(root)
        return [
            (flat_step_dir, f, category)
            for f in os.listdir(flat_step_dir)
            if _is_step_file(f)
        ]

    step_files = []
    for category in os.listdir(root):
        if category in {"processed", "checkpoints", "embeddings"}:
            continue
        step_dir = os.path.join(root, category, 'STEP', 'step final files')
        if not os.path.exists(step_dir):
            step_dir = os.path.join(root, category, 'STEP')
        if os.path.exists(step_dir):
            for f in os.listdir(step_dir):
                if _is_step_file(f):
                    step_files.append((step_dir, f, category))
    return step_files

class CADDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        
        processed_path = self.processed_paths[0]
        if os.path.exists(processed_path):
            self.data, self.slices = torch.load(processed_path, weights_only=False)
        else:
            legacy_path = os.path.join(self.processed_dir, "fabwave_data.pt")
            if os.path.exists(legacy_path):
                self.data, self.slices = torch.load(legacy_path, weights_only=False)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['graphs.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        step_files = _discover_step_files(self.root)
        
        print(f"Processing {len(step_files)} STEP files...")
        
        cpus = min(mp.cpu_count(), 8)
        if len(step_files) <= 1:
            cpus = 1
        print(f"Using {cpus} parallel workers")
        
        if cpus > 1:
            with mp.Pool(cpus) as pool:
                results = list(tqdm.tqdm(pool.imap(_process_single_file, step_files), total=len(step_files), desc="Generating graphs"))
        else:
            results = []
            for item in tqdm.tqdm(step_files, desc="Generating graphs"):
                results.append(_process_single_file(item))
        processed = [r for r in results if r is not None]
        data_list = [r["data"] if isinstance(r, dict) else r for r in processed]
        metadata = [r["metadata"] for r in processed if isinstance(r, dict)]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(metadata, os.path.join(self.processed_dir, "metadata.pt"))
        print(f"Saved {len(data_list)} graphs")


class FabWaveDataset(CADDataset):
    pass
