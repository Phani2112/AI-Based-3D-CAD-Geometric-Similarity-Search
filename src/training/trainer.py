import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.model.ebc_augmentation import EBCAugmentation
from src.model.contrastive_loss import InfoNCELoss

class CADGCLTrainer(pl.LightningModule):
    def __init__(self, model, lr=0.01):
        super().__init__()
        self.model = model
        self.lr = lr
        self.aug = EBCAugmentation(p_edge=0.1)
        self.loss_fn = InfoNCELoss(temperature=0.07)

    def training_step(self, batch, batch_idx):
        edge_index2, edge_attr2 = self.aug(batch.edge_index, batch.edge_attr)
        z1 = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        z2 = self.model(batch.x, edge_index2, edge_attr2, batch.batch)
        loss = self.loss_fn(z1, z2)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)