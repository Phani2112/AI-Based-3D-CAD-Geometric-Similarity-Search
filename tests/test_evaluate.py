from scripts.evaluate import average_precision_at_k, mean_average_precision_at_k


def test_average_precision_at_k_uses_standard_denominator():
    labels = ["a", "a", "b", "b"]

    assert average_precision_at_k(query_id=0, results=[2, 3, 1], labels=labels, k=3) == 1 / 3


def test_mean_average_precision_at_k_includes_zero_hit_queries():
    labels = ["a", "a", "b", "b"]
    search_results = {
        0: [1, 2],
        1: [2, 3],
        2: [3, 0],
        3: [0, 1],
    }

    assert mean_average_precision_at_k(labels, search_results, k=2) == 0.5


def test_evaluate_uses_dataset_artifact_paths():
    from src.data.dataset_registry import dataset_artifacts

    artifacts = dataset_artifacts("NewDataset")

    assert artifacts.metadata_path == "dataset/NewDataset/processed/metadata.pt"
    assert artifacts.model_path == "dataset/NewDataset/checkpoints/cadgcl_model.pt"
    assert artifacts.embeddings_path == "dataset/NewDataset/embeddings/embeddings.pt"
