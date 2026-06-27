import argparse
import torch
from torch_geometric.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Generate graph embeddings')
    parser.add_argument('--dataset', type=str, default='dataset/FabWave')
    parser.add_argument('--model', type=str, default='checkpoints/cadgcl_model.pt')
    parser.add_argument('--output', type=str, default='embeddings/fabwave.pt')
    args = parser.parse_args()
    
    from src.data.dataset_loader import FabWaveDataset
    from src.model.gnn_encoder import BAGConvEncoder
    
    dataset = FabWaveDataset(root=args.dataset)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    model = BAGConvEncoder()
    model.load_state_dict(torch.load(args.model, weights_only=False))
    model.eval()
    
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            emb = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_embeddings.append(emb)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings, args.output)
    print(f"Saved {len(embeddings)} embeddings to {args.output}")

if __name__ == '__main__':
    main()