def test_migrate_fabwave_artifacts_copies_without_deleting(tmp_path):
    import os
    import torch

    from scripts.migrate_fabwave_artifacts import migrate_fabwave_artifacts

    root = tmp_path
    os.makedirs(root / "dataset" / "FabWave" / "processed", exist_ok=True)
    os.makedirs(root / "checkpoints", exist_ok=True)
    os.makedirs(root / "embeddings", exist_ok=True)
    torch.save(("graphs", "slices"), root / "dataset" / "FabWave" / "processed" / "fabwave_data.pt")
    torch.save([{"uuid": "a"}], root / "dataset" / "FabWave" / "processed" / "fabwave_metadata.pt")
    torch.save({"w": 1}, root / "checkpoints" / "cadgcl_model.pt")
    torch.save("embeddings", root / "embeddings" / "fabwave.pt")

    migrate_fabwave_artifacts(str(root))

    assert (root / "dataset" / "FabWave" / "processed" / "fabwave_data.pt").exists()
    assert (root / "dataset" / "FabWave" / "processed" / "graphs.pt").exists()
    assert (root / "dataset" / "FabWave" / "processed" / "metadata.pt").exists()
    assert (root / "dataset" / "FabWave" / "checkpoints" / "cadgcl_model.pt").exists()
    assert (root / "dataset" / "FabWave" / "embeddings" / "embeddings.pt").exists()
