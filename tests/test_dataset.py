import cv2
import numpy as np
import pytest
import torch

from dataset import LeJEPADataset


@pytest.fixture(autouse=True)
def fake_split(monkeypatch) -> None:
    _, jpeg = cv2.imencode(".jpg", np.zeros((900, 1600, 3), dtype=np.uint8))

    class _Split:
        def cast_column(self, *args, **kwargs) -> "_Split":
            return self

        def __getitem__(self, i: int) -> dict:
            return {"image": {"bytes": jpeg.tobytes()}}

        def __len__(self) -> int:
            return 4

    monkeypatch.setattr(
        "dataset.dataset.load_dataset", lambda *args, **kwargs: _Split()
    )


def test_loader_returns_a_batch() -> None:
    batch_size = 2
    dataset = LeJEPADataset("train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    globals, locals = next(iter(loader))
    assert globals.shape == (batch_size, dataset.n_global, 3, 256, 256)
    assert locals.shape == (batch_size, dataset.n_local, 3, 112, 112)


def test_train_without_local_views() -> None:
    globals, locals = LeJEPADataset("train", n_global=4, n_local=0)[0]
    assert globals.shape == (4, 3, 256, 256)
    assert locals.numel() == 0


def test_other_splits_return_a_center_crop() -> None:
    assert LeJEPADataset("validation")[0].shape == (3, 384, 384)
