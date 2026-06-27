import torch
import os
import numpy as np

def compute_map_by_category():
    """Compute mAP using dataset categories as pseudo-labels."""
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
    
    # Load embeddings
    embeddings = torch.load('embeddings/fabwave.pt', weights_only=False)
    
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
    
    # Compute similarity matrix (sample for speed)
    sample_size = min(500, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    
    from src.search.predictor import CADGCLPredictor
    predictor = CADGCLPredictor('checkpoints/cadgcl_model.pt', 'embeddings/fabwave.pt')
    predictor.load()
    
    ap_scores = []
    for query_id in sample_indices[:50]:  # Test 50 queries
        results = predictor.search(query_id, k=100)  # Get more for AP@100
        query_uuid = id_to_uuid.get(query_id)
        query_cat = category_map.get(query_uuid, -1)
        
        if query_cat == -1:
            continue
        
        # Count correct (same category)
        correct = 0
        precisions = []
        for i, r in enumerate(results):
            r_uuid = id_to_uuid.get(r, '')
            r_cat = category_map.get(r_uuid, -1)
            if r_cat == query_cat:
                correct += 1
                precisions.append(correct / (i + 1))
        
        if correct > 0:
            ap_scores.append(np.mean(precisions) if precisions else 0)
    
    map_score = np.mean(ap_scores) if ap_scores else 0
    print(f"Estimated mAP@100 (on {len(ap_scores)} valid queries): {map_score:.4f}")
    return map_score

if __name__ == '__main__':
    compute_map_by_category()