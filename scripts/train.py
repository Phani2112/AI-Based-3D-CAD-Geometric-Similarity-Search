import argparse
import os
import torch
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from src.data.dataset_loader import FabWaveDataset
from src.model.gnn_encoder import BAGConvEncoder
from src.model.ebc_augmentation import EBCAugmentation
from src.model.contrastive_loss import InfoNCELoss

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
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    parser = argparse.ArgumentParser(description='Train CADGCL model')
    parser.add_argument('--dataset', type=str, default='dataset/FabWave')
    parser.add_argument('--output', type=str, default='checkpoints/cadgcl_model.pt')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    dataset = FabWaveDataset(root=args.dataset)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = CADGCLModel(lr=0.01)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto')
    
    trainer.fit(model, loader)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.encoder.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()