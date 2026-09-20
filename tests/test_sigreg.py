import torch

from lejepa import SIGReg

V, N, K = 4, 256, 16
SIGREG = SIGReg(num_slices=256)


def gaussian_data(seed: int = 0) -> torch.Tensor:
    return torch.randn(V, N, K, generator=torch.Generator().manual_seed(seed))


def compute_statistic(embedding: torch.Tensor, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return SIGREG(embedding)


def expected_isotropic_score() -> float:
    return ((1 - (-SIGREG.t.square()).exp()) @ SIGREG.weights).item()


def test_isotropic_matches_expected_score() -> None:
    scores = torch.stack(
        [compute_statistic(gaussian_data(s), seed=s) for s in range(16)]
    )
    stderr = scores.std() / len(scores) ** 0.5
    assert abs(scores.mean() - expected_isotropic_score()) < 5 * stderr


def test_degeneracies_exceed_isotropic_scores() -> None:
    x = gaussian_data()
    dead = x.clone()
    dead[..., 4:] = 0.0
    degenerate = {
        "12 of 16 dims dead": dead,
        "every sample identical": x[:, :1].repeat(1, N, 1),
        "wrong scale": 3 * x,
        "wrong mean": x + 2.0,
    }
    isotropic = [compute_statistic(gaussian_data(s), seed=s) for s in range(16)]
    ceiling = torch.stack(isotropic).max()
    for name, embedding in degenerate.items():
        assert compute_statistic(embedding) > ceiling, name


def test_gradient_is_finite_and_nonzero() -> None:
    x = gaussian_data().requires_grad_(True)
    compute_statistic(x).backward()
    grad = x.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().max() > 0
