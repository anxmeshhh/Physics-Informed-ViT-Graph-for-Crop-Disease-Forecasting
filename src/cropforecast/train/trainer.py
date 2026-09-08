"""Training loop for the full architecture.

Setting
-------
Training is **inductive**: a separate graph is built over each split, so message
passing never carries information from a test observation into a training one.
The transductive alternative (one graph over everything, with masks) is common in
the GNN literature and would score higher, but it lets test-node features
influence training representations, and we would rather report the defensible
number.

Each split is trained full-batch. The graphs are small enough for that to fit
comfortably in 6 GB, and it keeps the graph intact rather than shredding it into
mini-batches whose edges are mostly cut.

Objective
---------
    L = CE(disease) + a * GaussianNLL(risk) + b * CE(risk band) + lambda * L_physics

Risk regression uses a Gaussian negative log-likelihood rather than plain MSE so
that the predicted log-variance is trained to mean something - which is what
makes the monotonic-uncertainty physics constraint a real constraint rather than
a penalty on an arbitrary number.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..physics.losses import PhysicsWeights, physics_loss


@dataclass
class SplitTensors:
    """Everything one split needs, already on the target device."""

    vision: torch.Tensor
    climate: torch.Tensor
    meta: torch.Tensor
    physics: torch.Tensor
    labels: torch.Tensor
    risk_reg: torch.Tensor
    risk_cls: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_weight: torch.Tensor

    def to(self, device: torch.device) -> "SplitTensors":
        return SplitTensors(**{
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in self.__dict__.items()
        })

    @property
    def n(self) -> int:
        return self.labels.shape[0]


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lambda_physics: float = 0.3
    w_risk_reg: float = 1.0
    w_risk_cls: float = 0.5
    grad_clip: float = 1.0
    patience: int = 40
    log_every: int = 25
    physics_weights: PhysicsWeights = field(default_factory=PhysicsWeights)


def gaussian_nll(pred: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
                 ) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL, with log-variance clamped for stability."""
    logvar = logvar.clamp(-6.0, 3.0)
    return (0.5 * (logvar + (target - pred).pow(2) / logvar.exp())).mean()


def compute_losses(
    out: dict[str, torch.Tensor],
    batch: SplitTensors,
    cfg: TrainConfig,
    use_physics: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total objective plus a breakdown for logging."""
    ce = F.cross_entropy(out["class_logits"], batch.labels)
    nll = gaussian_nll(out["risk"], out["logvar"], batch.risk_reg)

    # Risk-band CE, flattened across horizons.
    b, h, lv = out["level_logits"].shape
    band = F.cross_entropy(out["level_logits"].reshape(b * h, lv),
                           batch.risk_cls.reshape(b * h))

    total = ce + cfg.w_risk_reg * nll + cfg.w_risk_cls * band
    parts = {"ce": float(ce.detach()), "risk_nll": float(nll.detach()),
             "risk_band_ce": float(band.detach())}

    if use_physics and cfg.lambda_physics > 0:
        phys, breakdown = physics_loss(
            out["risk"], out["logvar"], batch.physics,
            batch.edge_index, batch.edge_weight, cfg.physics_weights,
        )
        total = total + cfg.lambda_physics * phys
        parts["physics"] = float(phys.detach())
        parts.update({f"phys_{k}": v for k, v in breakdown.items()})

    return total, parts


@torch.no_grad()
def evaluate(model, batch: SplitTensors) -> dict[str, float]:
    """Accuracy, macro-F1, and per-horizon forecast quality."""
    model.eval()
    out = model(batch.vision, batch.climate, batch.meta,
                batch.edge_index, batch.edge_attr)

    pred = out["class_logits"].argmax(1)
    correct = (pred == batch.labels).float()
    acc = float(correct.mean())

    # Macro-F1 over the 38 classes, computed on-device.
    n_cls = out["class_logits"].shape[1]
    f1s = []
    for c in range(n_cls):
        tp = float(((pred == c) & (batch.labels == c)).sum())
        fp = float(((pred == c) & (batch.labels != c)).sum())
        fn = float(((pred != c) & (batch.labels == c)).sum())
        if tp + fp + fn > 0:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    metrics = {"acc": acc, "macro_f1": macro_f1}

    risk_pred, risk_true = out["risk"], batch.risk_reg
    for i in range(risk_true.shape[1]):
        err = risk_pred[:, i] - risk_true[:, i]
        ss_res = float((err ** 2).sum())
        ss_tot = float(((risk_true[:, i] - risk_true[:, i].mean()) ** 2).sum())
        metrics[f"risk_mae_h{i}"] = float(err.abs().mean())
        metrics[f"risk_r2_h{i}"] = 1.0 - ss_res / max(ss_tot, 1e-9)
        band_pred = out["level_logits"][:, i].argmax(-1)
        metrics[f"band_acc_h{i}"] = float((band_pred == batch.risk_cls[:, i]).float().mean())

    metrics["risk_r2_mean"] = float(np.mean(
        [metrics[f"risk_r2_h{i}"] for i in range(risk_true.shape[1])]))
    metrics["band_acc_mean"] = float(np.mean(
        [metrics[f"band_acc_h{i}"] for i in range(risk_true.shape[1])]))
    return metrics


def train_model(
    model,
    train: SplitTensors,
    val: SplitTensors,
    cfg: TrainConfig,
    device: torch.device,
    use_physics: bool = True,
    verbose: bool = True,
) -> tuple[torch.nn.Module, list[dict]]:
    """Full-batch training with early stopping on validation macro-F1."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history: list[dict] = []
    best_score, best_state, best_epoch = -np.inf, None, -1
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(train.vision, train.climate, train.meta,
                    train.edge_index, train.edge_attr)
        loss, parts = compute_losses(out, train, cfg, use_physics)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        val_metrics = evaluate(model, val)
        # Select on a blend of the two jobs the model has to do at once.
        score = val_metrics["macro_f1"] + val_metrics["risk_r2_mean"]

        row = {"epoch": epoch, "loss": float(loss.detach()), **parts,
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(f"    ep {epoch:4d}  loss {float(loss):7.4f}  "
                  f"val_acc {val_metrics['acc']:.4f}  "
                  f"val_F1 {val_metrics['macro_f1']:.4f}  "
                  f"val_riskR2 {val_metrics['risk_r2_mean']:+.4f}", flush=True)

        if epoch - best_epoch >= cfg.patience:
            if verbose:
                print(f"    early stop at epoch {epoch} "
                      f"(best {best_epoch}, score {best_score:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"    trained in {time.time()-t0:.1f}s, best epoch {best_epoch}")
    return model, history
