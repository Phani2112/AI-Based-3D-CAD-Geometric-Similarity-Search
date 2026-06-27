import torch
import torch.nn.functional as F

class CADGCLUI:
    def __init__(self, datasets):
        self.datasets = datasets
        self.embeddings = {}
    
    def load_embeddings(self, dataset_name, embeddings):
        """Load pre-computed graph embeddings for a dataset."""
        self.embeddings[dataset_name] = embeddings
    
    def search(self, query_model_id, dataset='fabwave'):
        """Find top-5 similar models using cosine similarity."""
        if dataset not in self.embeddings:
            return []
        
        query_emb = self.embeddings[dataset][query_model_id:query_model_id+1]
        all_embs = self.embeddings[dataset]
        
        similarities = F.cosine_similarity(query_emb, all_embs)
        top5_idx = similarities.topk(k=5, sorted=True)[1].tolist()
        
        return top5_idx