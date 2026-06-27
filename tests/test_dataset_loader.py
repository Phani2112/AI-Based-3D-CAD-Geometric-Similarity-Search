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