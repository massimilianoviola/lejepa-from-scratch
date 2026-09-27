import torch

from lejepa import SIGReg, ViT

IMG, PATCH, EMBED = 32, 8, 24


def make_vit(drop_path_rate: float = 0.0, rope_rescale_coords: float = 0.0) -> ViT:
    torch.manual_seed(0)
    return ViT(
        patch_size=PATCH,
        embed_dim=EMBED,
        depth=2,
        num_heads=3,
        drop_path_rate=drop_path_rate,
        rope_rescale_coords=rope_rescale_coords,
    ).eval()


def make_images(batch: int = 16, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, IMG, IMG, generator=generator)


def test_drop_path_is_stochastic_only_in_training() -> None:
    vit, x = make_vit(drop_path_rate=0.5), make_images()
    torch.manual_seed(0)
    assert torch.allclose(vit(x), vit(x))
    vit.train()
    torch.manual_seed(0)
    assert not torch.allclose(vit(x), vit(x))


def test_rope_jittering_is_stochastic_only_in_training() -> None:
    vit, x = make_vit(rope_rescale_coords=2.0), make_images()
    torch.manual_seed(0)
    assert torch.allclose(vit(x), vit(x))
    vit.train()
    torch.manual_seed(0)
    assert not torch.allclose(vit(x), vit(x))


def test_embeddings_reach_sigreg_gradients() -> None:
    vit, x = make_vit(), make_images().requires_grad_(True)
    torch.manual_seed(0)
    views = torch.stack([vit(x), vit(x.flip(-1))])  # (views, batch, dims)
    SIGReg(num_slices=32)(views).backward()
    grad = x.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().max() > 0
