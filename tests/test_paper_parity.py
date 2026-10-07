import torch


def test_bagconv_encoder_uses_two_layers_and_projection_head():
    from src.model.gnn_encoder import BAGConv, BAGConvEncoder

    encoder = BAGConvEncoder()

    conv_layers = [module for module in encoder.modules() if isinstance(module, BAGConv)]
    assert len(conv_layers) == 2
    assert hasattr(encoder, "projection_head")

    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.randn(4, 11)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    graph_embeddings = encoder(x, edge_index, edge_attr, batch)
    projected = encoder.project(graph_embeddings)

    assert graph_embeddings.shape == (2, 256)
    assert projected.shape == (2, 256)


def test_feature_masking_zeros_some_nodes_when_training():
    from src.model.feature_masking import FeatureMasking

    torch.manual_seed(1)
    masking = FeatureMasking(p_mask=0.5)
    masking.train()

    x = torch.ones(100, 16)
    masked = masking(x)

    zero_rows = (masked.abs().sum(dim=1) == 0).sum().item()
    assert 0 < zero_rows < x.shape[0]


def test_weighted_infonce_accepts_negative_weights():
    from src.model.contrastive_loss import InfoNCELoss

    loss_fn = InfoNCELoss(temperature=0.07)
    u = torch.randn(4, 256)
    v = torch.randn(4, 256)
    weights = torch.ones(4, 4)
    weights.fill_diagonal_(0)

    loss = loss_fn(u, v, negative_weights=weights)

    assert loss.item() > 0


def test_training_module_switches_to_bmm_after_warmup():
    from scripts.train import CADGCLModel

    model = CADGCLModel(bmm_start_epoch=5)
    assert model.should_use_bmm(epoch=4) is False
    assert model.should_use_bmm(epoch=5) is True
