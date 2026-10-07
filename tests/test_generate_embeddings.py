def test_generate_embeddings_uses_dataset_scoped_defaults():
    from src.data.dataset_registry import dataset_artifacts

    artifacts = dataset_artifacts("NewDataset")

    assert artifacts.model_path == "dataset/NewDataset/checkpoints/cadgcl_model.pt"
    assert artifacts.embeddings_path == "dataset/NewDataset/embeddings/embeddings.pt"
