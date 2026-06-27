import argparse
import os
import torch
import pytorch_lightning as pl
from torch_geometric.data import DataLoader

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
        self.example_input_array = None

    def training_step(self, batch, batch_idx):
        z1 = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        edge_index2, edge_attr2 = self.aug(batch.edge_index, batch.edge_attr)
        z2 = self.encoder(batch.x, edge_index2, edge_attr2, batch.batch)
        loss = self.loss_fn(z1, z2)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class TQDMProgressBar(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.bar = None
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.progress_bar_callback is None:
            n_batches = len(trainer.train_dataloader)
            if self.bar is None:
                from tqdm import tqdm
                self.bar = tqdm(total=n_batches * trainer.max_epochs, desc="Training")
            self.bar.update(1)
            self.bar.set_postfix(loss=trainer.callback_metrics.get('train_loss', 0))

def main():
    parser = argparse.ArgumentParser(description='Train CADGCL model')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Limit to N models for testing')
    args = parser.parse_args()
    
    os.makedirs('checkpoints', exist_ok=True)
    
    if args.quick:
        from torch_geometric.data import InMemoryDataset, Data
        class DemoDataset(InMemoryDataset):
            def __init__(self, num_graphs=100):
                super().__init__()
                data_list = []
                for _ in range(num_graphs):
                    n_nodes = torch.randint(5, 50, (1,)).item()
                    x = torch.randn(n_nodes, 16)
                    edge_index = torch.randint(0, n_nodes, (2, torch.randint(4, n_nodes*2, (1,)).item()))
                    edge_attr = torch.randn(edge_index.shape[1], 11)
                    data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
                self.data, self.slices = self.collate(data_list)
        print(f"Training on 100 synthetic graphs...")
        dataset = DemoDataset(num_graphs=100)
    else:
        from src.data.dataset_loader import FabWaveDataset
        print("Loading FabWave dataset...")
        dataset = FabWaveDataset(root="dataset/FabWave")
        
        if args.limit > 0:
            from torch.utils.data import Subset
            indices = list(range(min(args.limit, len(dataset))))
            dataset = Subset(dataset, indices)
            print(f"Using first {len(dataset)} models")
    
    print(f"Dataset has {len(dataset)} graphs")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(8, os.cpu_count() or 4))
    
    model = CADGCLModel(lr=0.01)
    
    # Enable progress bar
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        logger=False,
        enable_progress_bar=True,
        callbacks=[TQDMProgressBar()] if not hasattr(pl, 'callbacks') else []
    )
    
    print("Starting training...")
    trainer.fit(model, loader)
    
    torch.save(model.encoder.state_dict(), "checkpoints/cadgcl_model.pt")
    print(f"Model saved to checkpoints/cadgcl_model.pt")

if __name__ == '__main__':
    main()