import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetArtifacts:
    name: str
    root: str
    processed_dir: str
    graphs_path: str
    metadata_path: str
    checkpoint_dir: str
    model_path: str
    embeddings_dir: str
    embeddings_path: str


def dataset_artifacts(dataset: str, datasets_root: str = "dataset") -> DatasetArtifacts:
    if not dataset or os.path.basename(dataset) != dataset:
        raise ValueError("Dataset must be a folder name under the datasets root")
    root = os.path.join(datasets_root, dataset)
    processed_dir = os.path.join(root, "processed")
    checkpoint_dir = os.path.join(root, "checkpoints")
    embeddings_dir = os.path.join(root, "embeddings")
    return DatasetArtifacts(
        name=dataset,
        root=root,
        processed_dir=processed_dir,
        graphs_path=os.path.join(processed_dir, "graphs.pt"),
        metadata_path=os.path.join(processed_dir, "metadata.pt"),
        checkpoint_dir=checkpoint_dir,
        model_path=os.path.join(checkpoint_dir, "cadgcl_model.pt"),
        embeddings_dir=embeddings_dir,
        embeddings_path=os.path.join(embeddings_dir, "embeddings.pt"),
    )


def list_dataset_roots(datasets_root: str = "dataset") -> list[str]:
    if not os.path.isdir(datasets_root):
        return []
    names = []
    for name in os.listdir(datasets_root):
        path = os.path.join(datasets_root, name)
        if name == "processed" or not os.path.isdir(path):
            continue
        names.append(name)
    return sorted(names)


def list_ready_datasets(datasets_root: str = "dataset") -> list[str]:
    ready = []
    for name in list_dataset_roots(datasets_root):
        artifacts = dataset_artifacts(name, datasets_root)
        if (
            os.path.exists(artifacts.metadata_path)
            and os.path.exists(artifacts.model_path)
            and os.path.exists(artifacts.embeddings_path)
        ):
            ready.append(name)
    return ready
