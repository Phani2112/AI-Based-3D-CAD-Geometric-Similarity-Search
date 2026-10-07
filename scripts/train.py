import argparse
import os
import sys
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.gnn_encoder import BAGConvEncoder
from src.model.ebc_augmentation import EBCAugmentation
from src.model.contrastive_loss import InfoNCELoss
from src.model.feature_masking import FeatureMasking
from src.model.bmm_sampler import BMMSampler
from src.data.dataset_registry import dataset_artifacts


class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, num_graphs=100):
        from torch_geometric.data import Data

        self.graphs = []
        for _ in range(num_graphs):
            n_nodes = torch.randint(5, 50, (1,)).item()
            x = torch.randn(n_nodes, 16)
            edge_count = torch.randint(4, n_nodes * 2, (1,)).item()
            edge_index = torch.randint(0, n_nodes, (2, edge_count))
            edge_attr = torch.randn(edge_count, 11)
            self.graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

class CADGCLModel(pl.LightningModule):
    def __init__(self, lr=0.01, bmm_start_epoch=5):
        super().__init__()
        self.encoder = BAGConvEncoder()
        self.aug = EBCAugmentation(p_edge=0.1)
        self.feature_masking = FeatureMasking(p_mask=0.3)
        self.loss_fn = InfoNCELoss(temperature=0.07)
        self.bmm_sampler = BMMSampler()
        self.bmm_start_epoch = bmm_start_epoch
        self.lr = lr
        self.example_input_array = None

    def should_use_bmm(self, epoch: int) -> bool:
        return epoch >= self.bmm_start_epoch

    def _negative_weights(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        similarities = torch.nn.functional.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=2)
        similarities = ((similarities + 1) / 2).clamp(0, 1)
        flat_weights = self.bmm_sampler(similarities.flatten()).reshape_as(similarities)
        flat_weights = flat_weights.detach()
        flat_weights.fill_diagonal_(0)
        return flat_weights

    def training_step(self, batch, batch_idx):
        z1 = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        edge_index2, edge_attr2 = self.aug(batch.edge_index, batch.edge_attr)
        masked_x = self.feature_masking(batch.x)
        z2 = self.encoder(masked_x, edge_index2, edge_attr2, batch.batch)
        u = self.encoder.project(z1)
        v = self.encoder.project(z2)
        weights = self._negative_weights(u, v) if self.should_use_bmm(self.current_epoch) else None
        loss = self.loss_fn(u, v, negative_weights=weights)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class ProgressPrinter(pl.callbacks.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        total = trainer.num_training_batches
        if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total:
            loss = trainer.callback_metrics.get("train_loss")
            loss_text = f" loss={float(loss):.4f}" if loss is not None else ""
            print(
                f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
                f"batch {batch_idx + 1}/{total}{loss_text}",
                flush=True,
            )

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss_epoch") or trainer.callback_metrics.get("train_loss")
        loss_text = f" loss={float(loss):.4f}" if loss is not None else ""
        print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} finished{loss_text}", flush=True)

def main():
    parser = argparse.ArgumentParser(description='Train CADGCL model')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Limit to N models for testing')
    parser.add_argument('--skip-embeddings', action='store_true')
    parser.add_argument('--dataset', type=str, default='FabWave', help='Dataset folder name under dataset/')
    args = parser.parse_args()
    
    artifacts = dataset_artifacts(args.dataset)
    os.makedirs(artifacts.checkpoint_dir, exist_ok=True)
    os.makedirs(artifacts.embeddings_dir, exist_ok=True)
    
    if args.quick:
        print(f"Training on 100 synthetic graphs...")
        dataset = DemoDataset(num_graphs=100)
    else:
        from src.data.dataset_loader import CADDataset
        print(f"Loading {args.dataset} dataset from {artifacts.root}...")
        dataset = CADDataset(root=artifacts.root)
        
        if args.limit > 0:
            from torch.utils.data import Subset
            indices = list(range(min(args.limit, len(dataset))))
            dataset = Subset(dataset, indices)
            print(f"Using first {len(dataset)} models")
    
    print(f"Dataset has {len(dataset)} graphs")
    # Python 3.14 uses forkserver by default; PyTorch DataLoader workers can
    # exhaust file descriptors in this environment. Graph generation is already
    # parallelized and cached, so keep training data loading single-process.
    workers = 0
    print(f"Using DataLoader workers: {workers}", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers)
    
    model = CADGCLModel(lr=0.01, bmm_start_epoch=5)
    
    # Enable progress bar
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        logger=False,
        enable_progress_bar=True,
        callbacks=[ProgressPrinter()],
    )
    
    print("Starting training...")
    trainer.fit(model, loader)
    
    torch.save(model.encoder.state_dict(), artifacts.model_path)
    print(f"Model saved to {artifacts.model_path}")

    if not args.skip_embeddings:
        print("Generating embeddings for the trained model...", flush=True)
        embedding_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model.encoder.eval()
        embeddings = []
        with torch.no_grad():
            for batch in embedding_loader:
                embeddings.append(model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, artifacts.embeddings_path)
        print(f"Saved {len(embeddings)} embeddings to {artifacts.embeddings_path}", flush=True)

if __name__ == '__main__':
    main()
