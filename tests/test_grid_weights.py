# Third-party
import torch

# First-party
from neural_lam import metrics


def test_uniform_weights_match_unweighted():
    """With uniform weights (all 1s), output should match unweighted behavior."""
    B, N, d_state = 2, 10, 3
    pred = torch.randn(B, N, d_state)
    target = torch.randn(B, N, d_state)
    pred_std = torch.ones(d_state)
    mask = torch.ones(N, dtype=torch.bool)
    uniform_weights = torch.ones(N)

    for metric_func in (
        metrics.mse,
        metrics.mae,
        metrics.wmse,
        metrics.wmae,
        metrics.nll,
        metrics.crps_gauss,
    ):
        val_no_weights = metric_func(
            pred, target, pred_std, mask=mask, grid_weights=None
        )
        val_uniform = metric_func(
            pred, target, pred_std, mask=mask, grid_weights=uniform_weights
        )
        torch.testing.assert_close(val_no_weights, val_uniform)


def test_nonuniform_weights_differ():
    """With non-uniform weights, output should differ from unweighted."""
    B, N, d_state = 2, 10, 3
    pred = torch.randn(B, N, d_state)
    target = torch.randn(B, N, d_state)
    pred_std = torch.ones(d_state)
    mask = torch.ones(N, dtype=torch.bool)
    nonuniform_weights = torch.linspace(0.5, 2.0, N)

    for metric_func in (metrics.mse, metrics.mae):
        val_no_weights = metric_func(
            pred, target, pred_std, mask=mask, grid_weights=None
        )
        val_weighted = metric_func(
            pred, target, pred_std, mask=mask, grid_weights=nonuniform_weights
        )
        assert not torch.allclose(
            val_no_weights, val_weighted
        ), f"{metric_func.__name__} should differ with non-uniform weights"


def test_grid_weights_shape_broadcast():
    """Grid weights (N,) should broadcast correctly with (..., N, d_state)."""
    B, T, N, d_state = 2, 4, 10, 3
    pred = torch.randn(B, T, N, d_state)
    target = torch.randn(B, T, N, d_state)
    pred_std = torch.ones(d_state)
    weights = torch.ones(N) * 1.5

    # Should not raise, and output shape should be (B, T) with defaults
    result = metrics.mse(pred, target, pred_std, grid_weights=weights)
    assert result.shape == (B, T)

    # With sum_vars=False, shape should be (B, T, d_state)
    result_vars = metrics.mse(
        pred, target, pred_std, grid_weights=weights, sum_vars=False
    )
    assert result_vars.shape == (B, T, d_state)

    # With average_grid=False, shape should be (B, T, N)
    result_grid = metrics.mse(
        pred, target, pred_std, grid_weights=weights, average_grid=False
    )
    assert result_grid.shape == (B, T, N)
