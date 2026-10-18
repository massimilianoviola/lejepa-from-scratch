import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, DistributedSampler

from dataset import LeJEPADataset
from lejepa import SIGReg, ViT
from lejepa.projector import Projector
from utils.images import save_pca, save_views
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
    batch_size: int = 256
    num_workers: int = 4

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 5e-2
    epochs: int = 300

    # SIGReg
    num_slices: int = 1024
    knots: int = 17
    t_max: float = 5.0

    # Checkpoints
    run_name: str = "auto"
    save_every: int = 10


def main(cfg: Config) -> None:
    # Set by torchrun, plain `python train.py` falls back to a single device
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # bf16 on supported CUDA GPUs, fp16 on older CUDA GPUs, fp32 elsewhere
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported(including_emulation=False):
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    cfg.run_name = run_name(cfg.run_name)
    if rank == 0:
        # Only rank 0 logs and saves
        log_path = run_dir(cfg.run_name) / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w")
        print(f"log {log_path}", flush=True)

        # Fixed validation center crops, logged at 384
        eval_set = LeJEPADataset("validation", n_global=1, n_local=0)
        step = len(eval_set) // 8
        eval_images = torch.stack([eval_set[i * step][0] for i in range(8)])
        del eval_set
        save_views(eval_images, cfg.run_name)

    train_set = LeJEPADataset("train", n_global=cfg.n_global, n_local=cfg.n_local)
    sampler = DistributedSampler(train_set, drop_last=True) if world_size > 1 else None
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size // world_size,  # batch_size is the global batch
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    vit = ViT().to(device)
    projector = Projector().to(device)
    encoder = Encoder(vit, projector)
    if world_size > 1:
        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        encoder = DDP(encoder, broadcast_buffers=False)
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
        swa_lr=cfg.learning_rate / 10,
        anneal_epochs=swa_steps - warmup_steps,
        anneal_strategy="cos",
    )
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    for epoch in range(cfg.epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        encoder.train()
        if epoch == swa_start:
            swa.update_parameters(vit)
        totals = [0.0, 0.0, 0.0]
        for step, (global_views, local_views) in enumerate(loader, start=1):
            global_views = global_views.to(device, non_blocking=True)
            local_views = local_views.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(
                device.type, dtype=dtype, enabled=dtype != torch.float32
            ):
                bs = global_views.shape[0]
                # (batch, views, C, H, W) -> (batch * views, C, H, W)
                g_flat = global_views.flatten(0, 1)
                l_flat = local_views.flatten(0, 1)
                # (batch * views, dim) -> model -> (views, batch, dim)
                global_embedding = (
                    encoder(g_flat).view(bs, cfg.n_global, -1).transpose(0, 1)
                )
                local_embedding = (
                    encoder(l_flat).view(bs, cfg.n_local, -1).transpose(0, 1)
                )
                all_embedding = torch.cat([global_embedding, local_embedding], dim=0)
                centers = global_embedding.mean(0)
                sim = (centers - all_embedding).square().mean()
                reg = sigreg(all_embedding)
                loss = (1 - cfg.lambd) * sim + cfg.lambd * reg
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after >= scale_before:
                scheduler.step()

            if epoch >= swa_start:
                swa.update_parameters(vit)

            if rank != 0:
                continue
            values = [loss.item(), sim.item(), reg.item()]
            for i, value in enumerate(values):
                totals[i] += value
            print(
                f"epoch {epoch} | step {step} | loss {values[0]:.4f}"
                f" | sim {values[1]:.4f} | sigreg {values[2]:.4f}",
                file=log,
                flush=True,
            )
        if rank != 0:
            continue
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
            save_pca(tokens, cfg.run_name, f"epoch_{done:04d}")
            if done != cfg.epochs:
                # Once SWA has started, the averaged model is the one we save
                if epoch >= swa_start:
                    save_model(swa.module, cfg.run_name, f"epoch_{done:04d}_swa.pth")
                else:
                    save_model(vit, cfg.run_name, f"epoch_{done:04d}.pth")

    if rank == 0:
        save_model(swa.module, cfg.run_name, f"epoch_{cfg.epochs:04d}_final_swa.pth")
        log.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(Config())
