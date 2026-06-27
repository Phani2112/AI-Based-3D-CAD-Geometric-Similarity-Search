import argparse
import os
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, Data
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from src.model.gnn_encoder import BAGConvEncoder
from src.model.ebc_augmentation import EBCAugmentation
from src.model.contrastive_loss import InfoNCELoss

class DemoDataset(InMemoryDataset):
    def __init__(self, num_graphs=100):
        super().__init__()
        self.data, self.slices = self._generate_synthetic(num_graphs)
    
    def _generate_synthetic(self, num_graphs):
        data_list = []
        for _ in range(num_graphs):
            n_nodes = torch.randint(5, 50, (1,)).item()
            x = torch.randn(n_nodes, 16)
            edge_index = torch.randint(0, n_nodes, (2, torch.randint(4, n_nodes*2, (1,)).item()))
            edge_attr = torch.randn(edge_index.shape[1], 11)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)
        return self.collate(data_list)

class CADGCLModel(pl.LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        self.encoder = BAGConvEncoder()
        self.aug = EBCAugmentation(p_edge=0.1)
        self.loss_fn = InfoNCELoss(temperature=0.07)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        z1 = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        edge_index2, edge_attr2 = self.aug(batch.edge_index, batch.edge_attr)
        z2 = self.encoder(batch.x, edge_index2, edge_attr2, batch.batch)
        loss = self.loss_fn(z1, z2)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    parser = argparse.ArgumentParser(description='Train CADGCL model')
    parser.add_argument('--dataset', type=str, default='dataset/FabWave')
    parser.add_argument('--output', type=str, default='checkpoints/cadgcl_model.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--quick', action='store_true', help='Quick demo with synthetic data')
    parser.add_argument('--num-graphs', type=int, default=100, help='Synthetic graphs for demo')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.quick:
        print(f"Training on {args.num_graphs} synthetic graphs...")
        dataset = DemoDataset(num_graphs=args.num_graphs)
    else:
        print(f"Loading dataset from {args.dataset}...")
        from src.data.dataset_loader import FabWaveDataset
        dataset = FabWaveDataset(root=args.dataset)
        print(f"Dataset has {len(dataset)} graphs")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = CADGCLModel(lr=0.01)
    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        accelerator='auto',
        enable_progress_bar=True,
        logger=False
    )
    
    print("Starting training...")
    trainer.fit(model, loader)
    
    torch.save(model.encoder.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()