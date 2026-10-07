import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def average_precision_at_k(query_id: int, results: list[int], labels: list[str], k: int) -> float:
    query_label = labels[query_id]
    total_relevant = sum(1 for label in labels if label == query_label) - 1
    if total_relevant <= 0:
        return 0.0

    correct = 0
    precision_sum = 0.0
    filtered_results = [result for result in results if result != query_id][:k]
    for rank, result in enumerate(filtered_results, 1):
        if labels[result] == query_label:
            correct += 1
            precision_sum += correct / rank

    return precision_sum / min(k, total_relevant)


def mean_average_precision_at_k(labels: list[str], search_results: dict[int, list[int]], k: int) -> float:
    scores = [
        average_precision_at_k(query_id, search_results[query_id], labels, k)
        for query_id in range(len(labels))
    ]
    return float(np.mean(scores)) if scores else 0.0


def compute_map_by_category(dataset: str = "FabWave", k=10):
    """Compute mAP@K using categories aligned to processed graph/embedding rows."""
    from src.data.dataset_registry import dataset_artifacts
    from src.search.predictor import CADGCLPredictor

    artifacts = dataset_artifacts(dataset)
    metadata_path = artifacts.metadata_path
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Missing {metadata_path}. Reconstruct metadata before evaluating so labels align with embeddings."
        )

    metadata = torch.load(metadata_path, weights_only=False)
    labels = [item['category'] for item in metadata]

    predictor = CADGCLPredictor(artifacts.model_path, artifacts.embeddings_path)
    predictor.load()
    if len(labels) != len(predictor.embeddings):
        raise ValueError(
            f"Metadata count ({len(labels)}) does not match embedding count ({len(predictor.embeddings)})."
        )

    search_results = {
        query_id: predictor.search(query_id, k=k + 1)
        for query_id in range(len(labels))
    }
    map_score = mean_average_precision_at_k(labels, search_results, k)

    paper_target = 0.9858
    print(f"Queries evaluated: {len(labels)}")
    print(f"Categories: {len(Counter(labels))}")
    print(f"mAP@{k}: {map_score:.4f} ({map_score * 100:.2f}%)")
    print(f"Paper mAP@{k}: {paper_target:.4f} ({paper_target * 100:.2f}%)")
    print(f"Gap from paper: {paper_target - map_score:.4f} ({(paper_target - map_score) * 100:.2f} percentage points)")
    return map_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='FabWave')
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    compute_map_by_category(dataset=args.dataset, k=args.k)
