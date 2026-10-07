import albumentations as A
import numpy as np
import torch
from datasets import load_dataset

HF_REPO = "mviola/nuImages-cam-front"


def training_crop(
    size: int, scale: tuple[float, float], blur: float, solarize: bool = False
) -> A.Compose:
    """DINOv3 augmentation recipe: crop, color, blur, normalize."""
    steps = [
        A.RandomResizedCrop(size=(size, size), scale=scale),
        A.ColorJitter(
            brightness_range=(0.6, 1.4),
            contrast_range=(0.6, 1.4),
            saturation_range=(0.8, 1.2),
            hue_range=(-0.1, 0.1),
            p=0.8,
        ),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_range=(9, 9), sigma_range=(0.1, 2.0), p=blur),
    ]
    if solarize:
        steps.append(A.Solarize(p=0.2))
    steps += [A.Normalize(), A.ToTensorV2()]
    return A.Compose(steps)


class LeJEPADataset(torch.utils.data.Dataset):
    """Return local and global crops, augmented."""

    def __init__(self, split: str, n_global: int = 2, n_local: int = 6) -> None:
        assert n_global >= 1, "n_global must be at least 1"
        self.n_global = n_global
        self.n_local = n_local
        self.ds = load_dataset(HF_REPO, split=split)
        # Like DINOv3, first global always blurred, second rarely and solarized
        self.global1 = training_crop(256, (0.32, 1.0), blur=1.0)
        self.global2 = training_crop(256, (0.32, 1.0), blur=0.1, solarize=True)
        # While local crops are smaller and blurred half the time
        self.local = training_crop(112, (0.05, 0.32), blur=0.5)
        self.test = A.Compose(
            [
                A.SmallestMaxSize(max_size=384),
                A.CenterCrop(height=384, width=384),
                A.Normalize(),
                A.ToTensorV2(),
            ]
        )

    def __getitem__(self, i: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        img = np.asarray(self.ds[i]["image"].convert("RGB"))
        if self.n_local == 0:
            views = [self.test(image=img)["image"] for _ in range(self.n_global)]
            return torch.stack(views)
        globals_transform = (self.global1, self.global2)
        global_views = torch.stack(
            [globals_transform[v % 2](image=img)["image"] for v in range(self.n_global)]
        )
        local_views = torch.stack(
            [self.local(image=img)["image"] for _ in range(self.n_local)]
        )
        return global_views, local_views

    def __len__(self) -> int:
        return len(self.ds)
