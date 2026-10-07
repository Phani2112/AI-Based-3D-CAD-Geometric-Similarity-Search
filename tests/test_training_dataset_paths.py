def test_training_script_uses_dataset_scoped_artifacts():
    from src.data.dataset_registry import dataset_artifacts

    artifacts = dataset_artifacts("FabWave")

    assert artifacts.model_path == "dataset/FabWave/checkpoints/cadgcl_model.pt"
    assert artifacts.embeddings_path == "dataset/FabWave/embeddings/embeddings.pt"
