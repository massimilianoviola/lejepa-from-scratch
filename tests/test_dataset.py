import numpy as np
import torch
from PIL import Image

from dataset import LeJEPADataset


def test_loader_returns_a_batch(monkeypatch) -> None:
    image = Image.fromarray(np.zeros((900, 1600, 3), dtype=np.uint8))

    class _Split:
        def __getitem__(self, i: int) -> dict:
            return {"image": image}

        def __len__(self) -> int:
            return 4

    monkeypatch.setattr(
        "dataset.dataset.load_dataset", lambda *args, **kwargs: _Split()
    )
    batch_size = 2
    dataset = LeJEPADataset("train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    globals, locals = next(iter(loader))
    assert globals.shape == (batch_size, dataset.n_global, 3, 256, 256)
    assert locals.shape == (batch_size, dataset.n_local, 3, 112, 112)
