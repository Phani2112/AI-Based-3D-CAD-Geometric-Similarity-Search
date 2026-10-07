import torch
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None
from src.model.gnn_encoder import BAGConvEncoder
from src.data.step_parser import STEPParser
from src.data.graph_builder import BRepGraphBuilder

class CADGCLPredictor:
    def __init__(self, model_path, embeddings_path):
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.model = None
        self.index = None
        self.embeddings = None

    def load(self):
        self.model = BAGConvEncoder()
        self.model.load_state_dict(torch.load(self.model_path, weights_only=False), strict=False)
        self.model.eval()
        
        self.embeddings = torch.load(self.embeddings_path, weights_only=False)
        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = self.embeddings.detach().cpu().numpy().astype('float32')
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-12)
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)

    def search(self, query_model_id, k=5):
        if self.index is None:
            return []
        query = self.embeddings[query_model_id:query_model_id+1]
        scores, indices = self.index.search(query, k)
        return indices[0].tolist()

    def search_with_scores(self, query_model_id, k=5):
        if self.index is None:
            return []
        query = self.embeddings[query_model_id:query_model_id+1]
        scores, indices = self.index.search(query, k)
        return [(int(model_id), float(score)) for model_id, score in zip(indices[0], scores[0])]

    def predict(self, step_path):
        parser = STEPParser()
        builder = BRepGraphBuilder()
        shape = parser.parse(step_path)
        graph = builder.build(shape)
        
        self.model.eval()
        with torch.no_grad():
            emb = self.model(
                graph.x.unsqueeze(0),
                graph.edge_index,
                graph.edge_attr,
                torch.zeros(graph.x.shape[0], dtype=torch.long)
            )
        return emb.squeeze(0)
