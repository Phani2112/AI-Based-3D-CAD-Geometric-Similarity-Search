import os


def test_dataset_artifacts_are_scoped_under_dataset_folder(tmp_path):
    from src.data.dataset_registry import dataset_artifacts

    datasets_root = tmp_path / "dataset"
    artifacts = dataset_artifacts("NewDataset", datasets_root=str(datasets_root))

    assert artifacts.name == "NewDataset"
    assert artifacts.root == str(datasets_root / "NewDataset")
    assert artifacts.graphs_path == str(datasets_root / "NewDataset" / "processed" / "graphs.pt")
    assert artifacts.metadata_path == str(datasets_root / "NewDataset" / "processed" / "metadata.pt")
    assert artifacts.model_path == str(datasets_root / "NewDataset" / "checkpoints" / "cadgcl_model.pt")
    assert artifacts.embeddings_path == str(datasets_root / "NewDataset" / "embeddings" / "embeddings.pt")


def test_list_dataset_roots_ignores_processed_and_files(tmp_path):
    from src.data.dataset_registry import list_dataset_roots

    datasets_root = tmp_path / "dataset"
    (datasets_root / "FabWave").mkdir(parents=True)
    (datasets_root / "NewDataset").mkdir()
    (datasets_root / "processed").mkdir()
    (datasets_root / "README.txt").write_text("not a dataset")

    assert list_dataset_roots(str(datasets_root)) == ["FabWave", "NewDataset"]


def test_list_ready_datasets_requires_model_embeddings_and_metadata(tmp_path):
    from src.data.dataset_registry import dataset_artifacts, list_ready_datasets

    datasets_root = tmp_path / "dataset"
    fabwave = dataset_artifacts("FabWave", str(datasets_root))
    os.makedirs(fabwave.processed_dir, exist_ok=True)
    os.makedirs(fabwave.checkpoint_dir, exist_ok=True)
    os.makedirs(fabwave.embeddings_dir, exist_ok=True)
    for path in [fabwave.metadata_path, fabwave.model_path, fabwave.embeddings_path]:
        open(path, "wb").close()
    os.makedirs(datasets_root / "Incomplete", exist_ok=True)

    assert list_ready_datasets(str(datasets_root)) == ["FabWave"]
