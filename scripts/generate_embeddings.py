import argparse
import torch
from torch_geometric.loader import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Generate graph embeddings')
    parser.add_argument('--dataset', type=str, default='FabWave', help='Dataset folder name under dataset/')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    from src.data.dataset_loader import CADDataset
    from src.data.dataset_registry import dataset_artifacts
    from src.model.gnn_encoder import BAGConvEncoder

    artifacts = dataset_artifacts(args.dataset)
    model_path = args.model or artifacts.model_path
    output_path = args.output or artifacts.embeddings_path
    
    dataset = CADDataset(root=artifacts.root)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    model = BAGConvEncoder()
    model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
    model.eval()
    
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            emb = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_embeddings.append(emb)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings, output_path)
    print(f"Saved {len(embeddings)} embeddings to {output_path}")

if __name__ == '__main__':
    main()
