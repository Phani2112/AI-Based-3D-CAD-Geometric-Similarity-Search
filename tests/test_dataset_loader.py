import pytest
from src.data.dataset_loader import FabWaveDataset
import os
import shutil

def test_dataset_loading():
    test_root = "/tmp/test_fabwave"
    shutil.rmtree(test_root, ignore_errors=True)
    os.makedirs(f"{test_root}/Bolts/STEP", exist_ok=True)
    shutil.copy("dataset/FabWave/Bolts/STEP/07b46ed1-3801-45ad-9f42-5adfffb4e1c7.stp", f"{test_root}/Bolts/STEP/")
    
    dataset = FabWaveDataset(root=test_root)
    assert len(dataset) == 1
    data = dataset[0]
    assert data.x.shape[1] == 16
    assert data.edge_attr.shape[1] == 11

def test_dataset_loading_nested():
    test_root = "/tmp/test_fabwave2"
    shutil.rmtree(test_root, ignore_errors=True)
    os.makedirs(f"{test_root}/Bearings/STEP/step final files", exist_ok=True)
    shutil.copy("dataset/FabWave/Bearings/STEP/00ed2536-3d80-4f07-8851-4f49f1606498.stp", f"{test_root}/Bearings/STEP/step final files/")
    
    dataset = FabWaveDataset(root=test_root)
    assert len(dataset) == 1


def test_cad_dataset_writes_graphs_and_metadata(tmp_path, monkeypatch):
    import torch
    from torch_geometric.data import Data

    from src.data import dataset_loader
    from src.data.dataset_loader import CADDataset

    step_dir = tmp_path / "MyDataset" / "CategoryA" / "STEP"
    step_dir.mkdir(parents=True)
    step_path = step_dir / "part-a.stp"
    step_path.write_text("ISO-10303-21;")

    def fake_process_single_file(args):
        step_dir_arg, filename, category = args
        return {
            "data": Data(
                x=torch.ones(1, 16),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 11),
            ),
            "metadata": {
                "uuid": "part-a",
                "category": category,
                "filename": filename,
                "path": str(step_path),
            },
        }

    monkeypatch.setattr(dataset_loader, "_process_single_file", fake_process_single_file)
    dataset = CADDataset(root=str(tmp_path / "MyDataset"))

    assert len(dataset) == 1
    metadata = torch.load(tmp_path / "MyDataset" / "processed" / "metadata.pt", weights_only=False)
    assert metadata == [{"uuid": "part-a", "category": "CategoryA", "filename": "part-a.stp", "path": str(step_path)}]
    assert (tmp_path / "MyDataset" / "processed" / "graphs.pt").exists()


def test_cad_dataset_supports_flat_step_layout(tmp_path, monkeypatch):
    import torch
    from torch_geometric.data import Data

    from src.data import dataset_loader
    from src.data.dataset_loader import CADDataset

    step_dir = tmp_path / "datasetguhring" / "step"
    step_dir.mkdir(parents=True)
    step_path = step_dir / "part-a.stp"
    step_path.write_text("ISO-10303-21;")

    def fake_process_single_file(args):
        step_dir_arg, filename, category = args
        assert step_dir_arg == str(step_dir)
        assert filename == "part-a.stp"
        assert category == "datasetguhring"
        return {
            "data": Data(
                x=torch.ones(1, 16),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 11),
            ),
            "metadata": {
                "uuid": "part-a",
                "category": category,
                "filename": filename,
                "path": str(step_path),
            },
        }

    monkeypatch.setattr(dataset_loader, "_process_single_file", fake_process_single_file)
    dataset = CADDataset(root=str(tmp_path / "datasetguhring"))

    assert len(dataset) == 1
    metadata = torch.load(tmp_path / "datasetguhring" / "processed" / "metadata.pt", weights_only=False)
    assert metadata == [{"uuid": "part-a", "category": "datasetguhring", "filename": "part-a.stp", "path": str(step_path)}]
