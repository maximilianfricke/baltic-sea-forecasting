# Third-party
import pytest
import torch

# First-party
from neural_lam import utils
from neural_lam.interaction_net import (
    FlexibleNet,
    InteractionNet,
    PropagationNet,
)


def _graph(num_send=13, num_rec=11, num_edges=47, seed=0):
    generator = torch.Generator().manual_seed(seed)
    send = torch.randint(0, num_send, (num_edges,), generator=generator)
    # Receiver 0 gets no edges, to check mean aggregation over in-degree
    rec = torch.randint(1, num_rec, (num_edges,), generator=generator)
    return torch.stack((send, rec)), num_send, num_rec


def _assert_same(chunked, full):
    if isinstance(full, tuple):
        for a, b in zip(chunked, full):
            _assert_same(a, b)
        return
    torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cls", [InteractionNet, PropagationNet])
@pytest.mark.parametrize("update_edges", [True, False])
@pytest.mark.parametrize("rows", [1, 5, 46])
def test_chunked_interaction_net_matches(cls, update_edges, rows):
    torch.manual_seed(0)
    edge_index, num_send, num_rec = _graph()
    d = 8
    net = cls(edge_index, d, update_edges=update_edges, num_rec=num_rec)
    send = torch.randn(2, num_send, d)
    rec = torch.randn(2, num_rec, d)
    edges = torch.randn(2, edge_index.shape[1], d)
    with torch.no_grad():
        full = net(send, rec, edges)
        with utils.inference_chunking(rows):
            chunked = net(send, rec, edges)
    _assert_same(chunked, full)


@pytest.mark.parametrize("propagation", [True, False])
def test_chunked_flexible_net_matches(propagation):
    torch.manual_seed(1)
    edge_index, num_send, num_rec = _graph(seed=1)
    net = FlexibleNet(
        edge_index, 6, 5, 4, num_rec=num_rec, propagation=propagation
    )
    send = torch.randn(2, num_send, 6)
    rec = torch.randn(2, num_rec, 5)
    edges = torch.randn(2, edge_index.shape[1], 4)
    with torch.no_grad():
        full = net(send, rec, edges)
        with utils.inference_chunking(4):
            chunked = net(send, rec, edges)
    _assert_same(chunked, full)


@pytest.mark.parametrize("cond_dim", [None, 3])
def test_chunked_mlp_matches(cond_dim):
    torch.manual_seed(2)
    mlp = utils.make_mlp([5, 7, 6], cond_dim=cond_dim)
    x = torch.randn(2, 29, 5)
    # Conditioning shared by all rows of a sample
    emb = None if cond_dim is None else torch.randn(2, 1, cond_dim)
    with torch.no_grad():
        full = mlp(x, emb)
        with utils.inference_chunking(4):
            chunked = mlp(x, emb)
    _assert_same(chunked, full)


def test_cond_mlp_with_per_row_emb_is_not_chunked():
    torch.manual_seed(3)
    mlp = utils.make_mlp([5, 6], cond_dim=3)
    x, emb = torch.randn(2, 29, 5), torch.randn(2, 29, 3)
    with torch.no_grad():
        full = mlp(x, emb)
        with utils.inference_chunking(4):
            chunked = mlp(x, emb)
    _assert_same(chunked, full)


def test_chunking_is_off_with_gradients():
    edge_index, num_send, num_rec = _graph()
    net = InteractionNet(edge_index, 4, update_edges=False, num_rec=num_rec)
    send = torch.randn(num_send, 4, requires_grad=True)
    with utils.inference_chunking(3):
        assert utils.inference_chunk_rows() is None
        out = net(send, torch.randn(num_rec, 4), torch.randn(47, 4))
    out.sum().backward()
    assert send.grad is not None
    with torch.no_grad(), utils.inference_chunking(3):
        assert utils.inference_chunk_rows() == 3
    assert utils.inference_chunk_rows() is None
