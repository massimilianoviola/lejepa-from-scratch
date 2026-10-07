from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader

from dataset import LeJEPADataset
from lejepa import SIGReg, ViT
from lejepa.projector import Projector
from utils.images import save_images
from utils.run import run_dir, run_name, save_model


class Encoder(torch.nn.Module):
    """Wrapper so the ViT and the projector are a single module."""

    def __init__(self, vit: ViT, projector: Projector) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.vit(x))


@dataclass
class Config:
    # Loss
    lambd: float = 0.05

    # DataLoader
    n_global: int = 2
    n_local: int = 6
    batch_size: int = 192
    num_workers: int = 4

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 200

    # SIGReg
    num_slices: int = 1024
    knots: int = 17
    t_max: float = 5.0

    # Checkpoints
    run_name: str = "auto"
    save_every: int = 10


def main(cfg: Config) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # bf16 on CUDA devices that support it, fp32 elsewhere
    dtype = (
        torch.bfloat16
        if device.type == "cuda"
        and torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float32
    )

    cfg.run_name = run_name(cfg.run_name)
    log_path = run_dir(cfg.run_name) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    print(f"log {log_path}", flush=True)

    # Fixed validation center crops, logged at 384
    eval_set = LeJEPADataset("validation", n_global=1, n_local=0)
    step = len(eval_set) // 8
    eval_images = torch.stack([eval_set[i * step][0] for i in range(8)])
    del eval_set

    loader = DataLoader(
        LeJEPADataset("train", n_global=cfg.n_global, n_local=cfg.n_local),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    vit = ViT().to(device)
    projector = Projector().to(device)
    encoder = Encoder(vit, projector)
    # Split the batch across GPUs
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
    # Stochastic weight average of the ViT kept as final model
    swa = AveragedModel(vit)
    sigreg = SIGReg(knots=cfg.knots, t_max=cfg.t_max, num_slices=cfg.num_slices).to(
        device
    )

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Warmup for one epoch
    warmup_steps = len(loader)
    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    # Use the last quarter for SWA
    swa_start = int(cfg.epochs * 0.75)
    swa_steps = swa_start * len(loader)
    s2 = SWALR(
        optimizer,
        swa_lr=cfg.learning_rate / 50,
        anneal_epochs=swa_steps - warmup_steps,
        anneal_strategy="cos",
    )
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    for epoch in range(cfg.epochs):
        encoder.train()
        if epoch == swa_start:
            swa.update_parameters(vit)
        totals = [0.0, 0.0, 0.0]
        for step, (global_views, local_views) in enumerate(loader, start=1):
            global_views = global_views.to(device, non_blocking=True)
            local_views = local_views.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(
                device.type, dtype=dtype, enabled=dtype == torch.bfloat16
            ):
                bs = global_views.shape[0]
                # (batch, views, C, H, W) -> (views * batch, C, H, W)
                g_flat = global_views.transpose(0, 1).flatten(0, 1)
                l_flat = local_views.transpose(0, 1).flatten(0, 1)
                # (views * batch, dim) -> (views, batch, dim)
                global_embedding = encoder(g_flat).view(cfg.n_global, bs, -1)
                local_embedding = encoder(l_flat).view(cfg.n_local, bs, -1)
                all_embedding = torch.cat([global_embedding, local_embedding], dim=0)
                centers = global_embedding.mean(0)
                sim = (centers - all_embedding).square().mean()
                reg = sigreg(all_embedding)
                loss = (1 - cfg.lambd) * sim + cfg.lambd * reg
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch >= swa_start:
                swa.update_parameters(vit)

            values = [loss.item(), sim.item(), reg.item()]
            for i, value in enumerate(values):
                totals[i] += value
            print(
                f"epoch {epoch} | step {step} | loss {values[0]:.4f}"
                f" | sim {values[1]:.4f} | sigreg {values[2]:.4f}",
                file=log,
                flush=True,
            )
        steps = len(loader)
        line = (
            f"epoch {epoch} | loss {totals[0] / steps:.4f}"
            f" | sim {totals[1] / steps:.4f} | sigreg {totals[2] / steps:.4f}"
        )
        print(line, flush=True)
        print(line, file=log, flush=True)
        done = epoch + 1
        if done % cfg.save_every == 0:
            encoder.eval()
            with torch.inference_mode():
                # Patch tokens only, CLS is the first token
                tokens = vit.forward_features(eval_images.to(device, non_blocking=True))
                tokens = tokens[:, 1:]
            save_images(eval_images, tokens, cfg.run_name, f"epoch_{done:04d}")
            if done != cfg.epochs:
                # Once SWA has started, the averaged model is the one we save
                if epoch >= swa_start:
                    save_model(swa.module, cfg.run_name, f"epoch_{done:04d}_swa.pth")
                else:
                    save_model(vit, cfg.run_name, f"epoch_{done:04d}.pth")

    save_model(swa.module, cfg.run_name, f"epoch_{cfg.epochs:04d}_final_swa.pth")
    log.close()


if __name__ == "__main__":
    main(Config())
