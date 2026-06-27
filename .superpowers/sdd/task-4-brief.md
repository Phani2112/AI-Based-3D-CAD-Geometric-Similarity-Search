### Task 4: Dataset Loader

**Files:**
- Create: `src/data/dataset_loader.py`
- Test: `tests/test_dataset_loader.py`

**Interfaces:**
- Consumes: Dataset folder path
- Produces: PyTorch Geometric DataLoaders

**Important:** FabWave dataset has 4,571 STEP files in nested subfolders: `dataset/FabWave/<Category>/STEP/step final files/*.stp`

**Step 1: Write the failing test**

```python
def test_dataset_loading():
    from src.data.dataset_loader import FabWaveDataset
    
    dataset = FabWaveDataset(root="dataset/FabWave")
    assert len(dataset) == 4571
    data = dataset[0]
    assert data.x.shape[1] == 16
    assert data.edge_attr.shape[1] == 11
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_loader.py -v`
Expected: FAIL

**Step 3: Write minimal dataset implementation**

```python
# src/data/dataset_loader.py
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
        self.data, self.slices = torch.load(self.processed_paths[0])
    
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
            if os.path.exists(step_dir):
                for f in os.listdir(step_dir):
                    if f.endswith('.stp') or f.endswith('.step'):
                        shape = self.parser.parse(os.path.join(step_dir, f))
                        graph = self.builder.build(shape)
                        if graph is not None:
                            data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/dataset_loader.py tests/test_dataset_loader.py
git commit -m "feat: add FabWave dataset loader"
```

Report file: `/home/jose-draeger/workspace/CADGCL V2/.superpowers/sdd/task-4-report.md`