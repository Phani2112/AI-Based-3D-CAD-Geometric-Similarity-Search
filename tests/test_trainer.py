import torch
from src.training.trainer import CADGCLTrainer
from src.model.gnn_encoder import BAGConvEncoder

def test_trainer_step():
    try:
        import faiss
    except ImportError:
        return
    
    model = BAGConvEncoder()
    trainer = CADGCLTrainer(model, lr=0.01)
    
    class MockBatch:
        def __init__(self):
            self.x = torch.randn(10, 16)
            self.edge_index = torch.randint(0, 10, (2, 20))
            self.edge_attr = torch.randn(20, 11)
            self.batch = torch.tensor([0,0,0,0,0,1,1,1,1,1])
    
    batch = MockBatch()
    loss = trainer.training_step(batch, 0)
    assert loss is not None
    assert loss.item() > 0