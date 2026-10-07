import os
import shutil


def copy_if_missing(source: str, target: str):
    if os.path.exists(source) and not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)
        print(f"Copied {source} -> {target}")


def migrate_fabwave_artifacts(project_root: str = "."):
    copy_if_missing(
        os.path.join(project_root, "dataset", "FabWave", "processed", "fabwave_data.pt"),
        os.path.join(project_root, "dataset", "FabWave", "processed", "graphs.pt"),
    )
    copy_if_missing(
        os.path.join(project_root, "dataset", "FabWave", "processed", "fabwave_metadata.pt"),
        os.path.join(project_root, "dataset", "FabWave", "processed", "metadata.pt"),
    )
    copy_if_missing(
        os.path.join(project_root, "checkpoints", "cadgcl_model.pt"),
        os.path.join(project_root, "dataset", "FabWave", "checkpoints", "cadgcl_model.pt"),
    )
    copy_if_missing(
        os.path.join(project_root, "embeddings", "fabwave.pt"),
        os.path.join(project_root, "dataset", "FabWave", "embeddings", "embeddings.pt"),
    )


if __name__ == "__main__":
    migrate_fabwave_artifacts()
