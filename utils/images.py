import torch
from PIL import Image

from utils.run import run_dir


def denorm(view: torch.Tensor) -> Image.Image:
    """Undo ImageNet normalization and return an RGB image."""
    view = view.detach().float().cpu()
    mean = view.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = view.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    image = (view * std + mean).clamp(0, 1)
    return Image.fromarray((image.permute(1, 2, 0) * 255).byte().numpy())


def save_views(images: torch.Tensor, run: str) -> None:
    """Write each image under run/<run>/images/."""
    folder = run_dir(run) / "images"
    folder.mkdir(parents=True, exist_ok=True)
    for i, view in enumerate(images, start=1):
        path = folder / f"image_{i:02d}.jpg"
        denorm(view).save(path)


def pca_rgb(tokens: torch.Tensor) -> torch.Tensor:
    """Map patch tokens to RGB with one 3-component PCA over the batch."""
    n, patches, _ = tokens.shape
    grid = int(patches**0.5)
    points = tokens.detach().float().cpu().flatten(0, 1)
    points = points - points.mean(0)
    _, _, basis = torch.pca_lowrank(points, q=3)
    coords = (points @ basis).view(n, grid, grid, 3)
    coords = coords - coords.amin(dim=(0, 1, 2))
    coords = coords / coords.amax(dim=(0, 1, 2)).clamp_min(1e-6)
    return (coords * 255).byte()


def save_pca(embeddings: torch.Tensor, run: str, name: str) -> None:
    """Write one RGB PCA per image under run/<run>/images/."""
    rgbs = pca_rgb(embeddings)
    side = rgbs.shape[1] * 16
    folder = run_dir(run) / "images"
    folder.mkdir(parents=True, exist_ok=True)
    for i, rgb in enumerate(rgbs, start=1):
        path = folder / f"image_{i:02d}_{name}_pca.jpg"
        Image.fromarray(rgb.numpy()).resize(
            (side, side), Image.Resampling.BILINEAR
        ).save(path)
