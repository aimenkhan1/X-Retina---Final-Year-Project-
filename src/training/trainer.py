"""Trainer: LR scheduler, early stopping, validation-based checkpointing. Owner: Maheen (Step 2, Week 9).

One Trainer is used for every experiment, so baseline, Focal Loss, SMOTE and hybrid runs
differ only in the model or the loss, never in how they are trained or scored.

What it does each epoch:
1. Train on the train loader (mixed precision on GPU, gradient clipping).
2. Score the val loader with src.evaluation.metrics.compute_metrics.
3. Save `last.pt` (for resuming after a Colab disconnect) and, if val QWK improved, `best.pt`.
4. Step the cosine learning-rate schedule; stop early if val QWK has not improved for
   `early_stopping_patience` epochs.
Every epoch is also appended to `history.csv` in the run folder.

Usage:
    trainer = Trainer(model, loaders, cfg["train"], run_dir=PATHS["checkpoints"] / "baseline_b4")
    history = trainer.fit()                       # resumes automatically if last.pt exists
    test_metrics = trainer.evaluate(loaders["test"], weights="best")
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.evaluation.metrics import compute_metrics, format_metrics


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)       # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)          # older torch


class Trainer:
    def __init__(self, model, loaders: dict, cfg: dict, run_dir, loss_fn=None, device=None,
                 max_batches=None, verbose=True):
        """
        cfg: the `train` block of configs/baseline.yaml (epochs, lr, weight_decay, scheduler,
             early_stopping_patience, mixed_precision, monitor).
        loss_fn: default cross-entropy. Step 4 passes Focal Loss here.
        max_batches: limit batches per epoch (smoke tests only).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loaders = loaders
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.max_batches = max_batches
        self.verbose = verbose

        self.epochs = int(cfg.get("epochs", 30))
        self.monitor = cfg.get("monitor", "qwk")              # higher is better
        self.patience = int(cfg.get("early_stopping_patience", 7))
        self.use_amp = bool(cfg.get("mixed_precision", True)) and self.device == "cuda"
        self.grad_clip = float(cfg.get("grad_clip", 1.0))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(cfg.get("lr", 1e-4)),
                                           weight_decay=float(cfg.get("weight_decay", 1e-4)))
        if cfg.get("scheduler", "cosine") == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        else:
            self.scheduler = None
        self.scaler = _grad_scaler(self.use_amp)

        self.start_epoch = 0
        self.best_score = -np.inf
        self.bad_epochs = 0
        self.history = []

    # ------------------------------------------------------------ checkpoints
    def _state(self, epoch):
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "best_score": self.best_score,
            "bad_epochs": self.bad_epochs,
            "history": self.history,
            "cfg": self.cfg,
        }

    def _save(self, name, epoch):
        """Write to a temp file first, then rename, so a disconnect never leaves a broken file."""
        path = self.run_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(self._state(epoch), tmp)
        os.replace(tmp, path)

    def resume(self) -> bool:
        """Load last.pt if it exists. Returns True if training continues from it."""
        path = self.run_dir / "last.pt"
        if not path.exists():
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if self.scheduler and ck.get("scheduler"):
            self.scheduler.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.start_epoch = ck["epoch"] + 1
        self.best_score = ck["best_score"]
        self.bad_epochs = ck["bad_epochs"]
        self.history = ck["history"]
        if self.verbose:
            print(f"Resumed from {path} (finished epoch {ck['epoch'] + 1}, best val {self.monitor} = {self.best_score:.4f})")
        return True

    def load_weights(self, name="best.pt"):
        ck = torch.load(self.run_dir / name, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        return ck

    # ------------------------------------------------------------ one pass
    def train_one_epoch(self, epoch):
        self.model.train()
        total, n = 0.0, 0
        loader = self.loaders["train"]
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{self.epochs} train", leave=False, disable=not self.verbose)
        for b, (x, y) in enumerate(bar):
            if self.max_batches and b >= self.max_batches:
                break
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                loss = self.loss_fn(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item() * len(y)
            n += len(y)
            bar.set_postfix(loss=f"{total / n:.4f}")
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, loader):
        """Returns (true labels, predicted labels, softmax probabilities, mean loss)."""
        self.model.eval()
        ys, ps, probs, total, n = [], [], [], 0.0, 0
        for b, (x, y) in enumerate(tqdm(loader, desc="eval", leave=False, disable=not self.verbose)):
            if self.max_batches and b >= self.max_batches:
                break
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
            p = torch.softmax(logits.float(), dim=1)
            total += loss.item() * len(y)
            n += len(y)
            ys.append(y.cpu().numpy())
            ps.append(p.argmax(1).cpu().numpy())
            probs.append(p.cpu().numpy())
        return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs), total / max(n, 1)

    def evaluate(self, loader, weights=None):
        """Metrics on any loader. weights='best' loads best.pt first."""
        if weights:
            self.load_weights(weights if weights.endswith(".pt") else f"{weights}.pt")
        y, p, _, loss = self.predict(loader)
        m = compute_metrics(y, p)
        m["loss"] = loss
        return m

    # ------------------------------------------------------------ main loop
    def fit(self, resume=True):
        if resume:
            self.resume()
        if self.start_epoch >= self.epochs:
            print("Already trained for all epochs. Increase `epochs` to continue.")
            return pd.DataFrame(self.history)
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_loss = self.train_one_epoch(epoch)
            y, p, _, val_loss = self.predict(self.loaders["val"])
            m = compute_metrics(y, p)
            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            score = m[self.monitor]
            improved = score > self.best_score
            if improved:
                self.best_score, self.bad_epochs = score, 0
            else:
                self.bad_epochs += 1

            row = {"epoch": epoch + 1, "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
                   "val_accuracy": m["accuracy"], "val_qwk": m["qwk"], "val_macro_f1": m["macro_f1"],
                   **{f"val_recall_{i}": r for i, r in enumerate(m["per_class_recall"])},
                   "best": improved, "seconds": round(time.time() - t0, 1)}
            self.history.append(row)

            if improved:
                self._save("best.pt", epoch)
            self._save("last.pt", epoch)                      # every epoch: needed for resume
            pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)

            if self.verbose:
                print(f"epoch {epoch + 1:>2}/{self.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                      f"val acc {m['accuracy']:.4f} | val QWK {m['qwk']:.4f} | "
                      f"{'BEST, saved' if improved else f'no gain {self.bad_epochs}/{self.patience}'} | {row['seconds']:.0f} s")

            if self.bad_epochs >= self.patience:
                print(f"Early stopping: val {self.monitor} did not improve for {self.patience} epochs.")
                break

        best = max(self.history, key=lambda r: r[f"val_{self.monitor}"])
        (self.run_dir / "summary.json").write_text(json.dumps(
            {"best_epoch": best["epoch"], f"best_val_{self.monitor}": best[f"val_{self.monitor}"],
             "epochs_run": len(self.history), "cfg": self.cfg}, indent=2, default=str))
        return pd.DataFrame(self.history)


def save_metrics(m: dict, path):
    """Save a metrics dict as JSON (for the results table in the report)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=2))
    return format_metrics(m)
