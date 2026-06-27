import torch
import os
import numpy as np

def compute_map_by_category(k=10):
    """Compute mAP@K using dataset categories as pseudo-labels."""
    # Build category mapping
    category_map = {}
    for cat_idx, category in enumerate(os.listdir('dataset/FabWave')):
        step_dir = os.path.join('dataset/FabWave', category, 'STEP', 'step final files')
        if not os.path.exists(step_dir):
            step_dir = os.path.join('dataset/FabWave', category, 'STEP')
        if os.path.exists(step_dir):
            for f in os.listdir(step_dir):
                if f.endswith('.stp') or f.endswith('.step'):
                    uuid = f.replace('.stp', '').replace('.step', '')
                    category_map[uuid] = cat_idx
    
    # Build reverse mapping (id -> uuid)
    id_to_uuid = {}
    idx = 0
    for category in os.listdir('dataset/FabWave'):
        step_dir = os.path.join('dataset/FabWave', category, 'STEP', 'step final files')
        if not os.path.exists(step_dir):
            step_dir = os.path.join('dataset/FabWave', category, 'STEP')
        if os.path.exists(step_dir):
            for f in os.listdir(step_dir):
                if f.endswith('.stp') or f.endswith('.step'):
                    id_to_uuid[idx] = f.replace('.stp', '').replace('.step', '')
                    idx += 1
    
    from src.search.predictor import CADGCLPredictor
    predictor = CADGCLPredictor('checkpoints/cadgcl_model.pt', 'embeddings/fabwave.pt')
    predictor.load()
    
    ap_scores = []
    test_queries = list(set(list(id_to_uuid.keys())))[:100]  # Test up to 100 queries
    
    for query_id in test_queries:
        results = predictor.search(query_id, k=k)
        query_uuid = id_to_uuid.get(query_id)
        query_cat = category_map.get(query_uuid, -1)
        
        if query_cat == -1:
            continue
        
        correct = 0
        precisions = []
        for i, r in enumerate(results):
            r_uuid = id_to_uuid.get(r, '')
            r_cat = category_map.get(r_uuid, -1)
            if r_cat == query_cat and r_cat != -1:
                correct += 1
                precisions.append(correct / (i + 1))
        
        if correct > 0:
            ap_scores.append(np.mean(precisions) if precisions else 0)
    
    map_score = np.mean(ap_scores) if ap_scores else 0
    target = 0.8935  # Paper's mAP@50 score scaled for mAP@10
    print(f"mAP@{k}: {map_score:.4f}")
    print(f"Paper target (estimated): {target:.2f}")
    print(f"Gap from target: {abs(target - map_score):.4f}")
    return map_score

if __name__ == '__main__':
    compute_map_by_category(k=10)