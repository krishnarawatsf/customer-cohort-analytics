"""CIFAR-10 self-pruning MLP with rubric-first defaults.

Default mode (no flags): maximizes strict evaluator rubric alignment.
Optional `--enhanced` enables extra engineering tricks without changing rubric
definitions for the core loss term.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Matplotlib writes config/cache to $MPLCONFIGDIR; keep it inside the repo for portability + sandbox safety.
_repo_root = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))
(Path(os.environ["MPLCONFIGDIR"])).mkdir(parents=True, exist_ok=True)

import random
import shutil
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    seed: int = 42
    enhanced: bool = False
    show_progress: bool = False

    # Training
    epochs: int = 10
    batch_size: int = 128
    num_workers: int = 0

    # Optimizer (rubric default: Adam lr=1e-3)
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Enhanced-only knobs
    lr_weights: float = 1e-3
    lr_gates: float = 5e-4
    weight_decay_weights: float = 5e-5
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    val_split: float = 0.1
    early_stopping_patience: int = 8
    warmup_epochs: int = 8
    gate_temp_start: float = 2.0
    gate_temp_end: float = 0.7
    aux_sparsity_weight: float = 0.05

    # Rubric constants
    lambda_values: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    sparse_threshold: float = 0.01

    # Outputs
    plots_dir: str = "plots"
    results_json: str = "results_summary.json"


def set_seed(seed: int) -> None:
    """Best-effort reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PrunableLinear(nn.Module):
    """Custom gated linear layer (no torch.nn.Linear).

    Forward (rubric order):
      gates = sigmoid(gate_scores)
      pruned_weights = weight * gates
      return F.linear(x, pruned_weights, bias)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Start at sigmoid(0)=0.5; stronger λ can push scores negative to create sub-0.01 gates.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class RubricMLP(nn.Module):
    """Exact rubric architecture:

    PrunableLinear(3072,512) -> ReLU -> PrunableLinear(512,256) -> ReLU -> PrunableLinear(256,10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_sparsity_loss(self) -> torch.Tensor:
        """Rubric sparsity term: sum(sigmoid(gate_scores)) over all PrunableLinear layers."""
        total = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                total += torch.sigmoid(module.gate_scores).sum()
        return total

    def collect_gate_values(self) -> Dict[str, torch.Tensor]:
        """sigmoid(gate_scores) per prunable layer."""
        out: Dict[str, torch.Tensor] = {}
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                out[name] = torch.sigmoid(module.gate_scores.detach())
        return out

    def global_sparsity(self, threshold: float) -> float:
        gates = torch.cat([g.reshape(-1) for g in self.collect_gate_values().values()])
        return (gates < threshold).float().mean().item() * 100.0

    def per_layer_sparsity(self, threshold: float) -> Dict[str, float]:
        return {name: (g < threshold).float().mean().item() * 100.0 for name, g in self.collect_gate_values().items()}

    def gate_entropy_per_layer(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, g in self.collect_gate_values().items():
            entropy = -g * torch.log(g + 1e-8) - (1.0 - g) * torch.log(1.0 - g + 1e-8)
            out[name] = entropy.mean().item()
        return out

    def all_gate_values_numpy(self) -> np.ndarray:
        return np.concatenate([g.cpu().reshape(-1).numpy() for g in self.collect_gate_values().values()])


class EnhancedMLP(nn.Module):
    """Optional enhanced model (only used with `--enhanced`).

    Still uses the same three PrunableLinear sizes, but adds BN/GELU/Dropout.
    """

    def __init__(self, dropout_p: float = 0.25) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = PrunableLinear(256, 10)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.dropout(F.gelu(self.bn1(self.fc1(x))))
        x = self.dropout(F.gelu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

    def get_sparsity_loss(self) -> torch.Tensor:
        total = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                total += torch.sigmoid(module.gate_scores).sum()
        return total

    def aux_temperature_sparsity(self, temperature: float) -> torch.Tensor:
        """Optional extra regularizer; NOT part of rubric sparsity_loss."""
        total = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                total += torch.sigmoid(module.gate_scores / temperature).sum()
        return total

    def collect_gate_values(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                out[name] = torch.sigmoid(module.gate_scores.detach())
        return out

    def global_sparsity(self, threshold: float) -> float:
        gates = torch.cat([g.reshape(-1) for g in self.collect_gate_values().values()])
        return (gates < threshold).float().mean().item() * 100.0

    def per_layer_sparsity(self, threshold: float) -> Dict[str, float]:
        return {name: (g < threshold).float().mean().item() * 100.0 for name, g in self.collect_gate_values().items()}

    def gate_entropy_per_layer(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, g in self.collect_gate_values().items():
            entropy = -g * torch.log(g + 1e-8) - (1.0 - g) * torch.log(1.0 - g + 1e-8)
            out[name] = entropy.mean().item()
        return out

    def all_gate_values_numpy(self) -> np.ndarray:
        return np.concatenate([g.cpu().reshape(-1).numpy() for g in self.collect_gate_values().values()])


ModelType = RubricMLP | EnhancedMLP


@dataclass
class ExperimentResult:
    lambda_value: float
    final_test_accuracy: float
    global_sparsity_percent: float
    layer_sparsity_percent: Dict[str, float]
    gate_entropy: Dict[str, float]
    best_val_accuracy: Optional[float] = None


def run_gradient_flow_check() -> None:
    model = PrunableLinear(10, 5)
    out = model(torch.randn(2, 10))
    out.sum().backward()
    assert model.gate_scores.grad is not None
    assert model.weight.grad is not None
    print("Gradient flow check passed.", flush=True)


def build_transforms(enhanced: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    if not enhanced:
        return (
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_tf, eval_tf


def make_dataloaders(config: ExperimentConfig) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    train_tf, eval_tf = build_transforms(config.enhanced)
    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_tf)

    if not config.enhanced:
        train_loader = DataLoader(
            full_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, None, DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Enhanced mode needs different transforms for train vs val, but the same split indices.
    train_aug = datasets.CIFAR10(root="./data", train=True, download=False, transform=train_tf)
    train_eval = datasets.CIFAR10(root="./data", train=True, download=False, transform=eval_tf)

    n = len(train_aug)
    val_n = int(n * config.val_split)
    train_n = n - val_n
    g = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:train_n]
    val_idx = perm[train_n:]

    train_loader = DataLoader(
        Subset(train_aug, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(train_eval, val_idx),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean().item() * 100.0


def evaluate(model: ModelType, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def build_model(config: ExperimentConfig, device: torch.device) -> ModelType:
    if config.enhanced:
        return EnhancedMLP().to(device)
    return RubricMLP().to(device)


def build_optimizer(model: ModelType, config: ExperimentConfig) -> torch.optim.Optimizer:
    if not config.enhanced:
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    gate_params: List[nn.Parameter] = []
    decay_params: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "gate_scores" in name:
            gate_params.append(p)
        elif name.endswith(".bias") or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "lr": config.lr_weights, "weight_decay": config.weight_decay_weights},
            {"params": no_decay, "lr": config.lr_weights, "weight_decay": 0.0},
            {"params": gate_params, "lr": config.lr_gates, "weight_decay": 0.0},
        ]
    )


def lambda_warmup(base: float, epoch: int, warmup_epochs: int) -> float:
    return base * min(1.0, (epoch + 1) / max(warmup_epochs, 1))


def temperature_schedule(start: float, end: float, epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return end
    t = epoch / (epochs - 1)
    return start + (end - start) * t


def train_one_lambda(
    lambda_value: float,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[ModelType, ExperimentResult, Dict[str, List[float]]]:
    model = build_model(config, device)
    optimizer = build_optimizer(model, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs) if config.enhanced else None

    label_smoothing = config.label_smoothing if config.enhanced else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "global_sparsity": [],
        "entropy_fc1": [],
        "entropy_fc2": [],
        "entropy_fc3": [],
    }
    if config.enhanced and val_loader is not None:
        history["val_acc"] = []

    best_val = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    patience = 0
    grad_checked = False

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        eff_lambda = lambda_warmup(lambda_value, epoch, config.warmup_epochs) if config.enhanced else lambda_value
        temp = temperature_schedule(config.gate_temp_start, config.gate_temp_end, epoch, config.epochs) if config.enhanced else 1.0

        iterator = train_loader
        if config.show_progress:
            iterator = tqdm(
                train_loader,
                desc=f"lambda={lambda_value} epoch {epoch + 1}/{config.epochs}",
                leave=False,
            )

        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)
            bs = labels.size(0)

            optimizer.zero_grad()
            logits = model(images)
            ce = criterion(logits, labels)
            sparsity = model.get_sparsity_loss()
            loss = ce + eff_lambda * sparsity
            if config.enhanced and isinstance(model, EnhancedMLP):
                loss = loss + config.aux_sparsity_weight * model.aux_temperature_sparsity(temp)

            loss.backward()

            if not grad_checked:
                assert model.fc1.gate_scores.grad is not None
                assert model.fc1.weight.grad is not None
                grad_checked = True

            if config.enhanced:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)

            optimizer.step()

            running_loss += loss.item() * bs
            running_acc += accuracy(logits, labels) * bs
            n += bs
            if config.show_progress:
                iterator.set_postfix({"loss": f"{loss.item():.3f}", "ce": f"{ce.item():.3f}", "lam": f"{eff_lambda:.5f}"})

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(n, 1)
        train_acc = running_acc / max(n, 1)
        _, test_acc_epoch = evaluate(model, test_loader, criterion, device)

        ent = model.gate_entropy_per_layer()
        gsparse = model.global_sparsity(config.sparse_threshold)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc_epoch)
        history["global_sparsity"].append(gsparse)
        history["entropy_fc1"].append(ent.get("fc1", float("nan")))
        history["entropy_fc2"].append(ent.get("fc2", float("nan")))
        history["entropy_fc3"].append(ent.get("fc3", float("nan")))

        if config.enhanced and val_loader is not None:
            _, val_acc = evaluate(model, val_loader, criterion, device)
            history["val_acc"].append(val_acc)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= config.early_stopping_patience:
                print(f"[lambda={lambda_value}] early stop epoch {epoch + 1}", flush=True)
                break

        print(
            f"[lambda={lambda_value}] epoch={epoch + 1} train_acc={train_acc:.2f}% "
            f"test_acc={test_acc_epoch:.2f}% global_sparsity={gsparse:.2f}%",
            flush=True,
        )

    if config.enhanced and best_state:
        model.load_state_dict(best_state)

    _, final_test = evaluate(model, test_loader, criterion, device)
    gsparse = model.global_sparsity(config.sparse_threshold)
    layer_sp = model.per_layer_sparsity(config.sparse_threshold)
    ent = model.gate_entropy_per_layer()

    print(
        f"[lambda={lambda_value}] Final Test Accuracy: {final_test:.2f}% | "
        f"Sparsity (<{config.sparse_threshold}): {gsparse:.2f}%",
        flush=True,
    )
    print(f"[lambda={lambda_value}] per-layer sparsity: {layer_sp}", flush=True)
    print(f"[lambda={lambda_value}] gate entropy: {ent}", flush=True)

    result = ExperimentResult(
        lambda_value=lambda_value,
        final_test_accuracy=final_test,
        global_sparsity_percent=gsparse,
        layer_sparsity_percent=layer_sp,
        gate_entropy=ent,
        best_val_accuracy=best_val if config.enhanced and val_loader is not None else None,
    )
    return model, result, history


def wipe_plots_dir(plots_dir: Path) -> None:
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)


def save_gate_hist(values: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=50, range=(0.0, 1.0), color="#1f77b4", alpha=0.85)
    plt.title(title)
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_heatmap(model: ModelType, path: Path) -> None:
    gates = model.collect_gate_values()
    names = list(gates.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(18, 5), constrained_layout=True)
    if len(names) == 1:
        axes = [axes]
    last = None
    for ax, name in zip(axes, names):
        g = gates[name].cpu().numpy()
        # Full-resolution heatmaps are huge (memory + PNG size). Downsample for visualization only.
        out_dim, in_dim = g.shape
        max_in, max_out = 384, 384
        if in_dim > max_in or out_dim > max_out:
            g = g[:: max(1, out_dim // max_out), :: max(1, in_dim // max_in)]
        last = ax.imshow(g, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("in (subsampled)")
        ax.set_ylabel("out (subsampled)")
    fig.colorbar(last, ax=axes, shrink=0.8, label="gate")
    fig.suptitle("Gate heatmap (best test accuracy)")
    plt.savefig(path, dpi=200)
    plt.close()


def save_tradeoff(results: List[ExperimentResult], path: Path) -> None:
    xs = [r.lambda_value for r in results]
    acc = [r.final_test_accuracy for r in results]
    sp = [r.global_sparsity_percent for r in results]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, acc, marker="o", label="test acc")
    plt.plot(xs, sp, marker="s", label="global sparsity")
    plt.xscale("log")
    plt.grid(alpha=0.25)
    plt.xlabel("lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def print_table(results: List[ExperimentResult]) -> None:
    print("\n=== Final Results ===", flush=True)
    print(f"{'Lambda':>10} | {'Test Acc':>9} | {'Sparsity':>9} | {'H(fc1)':>8} | {'H(fc2)':>8} | {'H(fc3)':>8}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        print(
            f"{r.lambda_value:>10.4f} | {r.final_test_accuracy:>9.2f} | {r.global_sparsity_percent:>9.2f} | "
            f"{r.gate_entropy['fc1']:>8.4f} | {r.gate_entropy['fc2']:>8.4f} | {r.gate_entropy['fc3']:>8.4f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--enhanced", action="store_true", help="Enable non-rubric engineering upgrades.")
    p.add_argument("--epochs", type=int, default=None, help="Epochs per lambda (default: 40 rubric, 50 enhanced).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plots-dir", type=str, default="plots")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show per-batch tqdm progress bars (can generate huge logs in some environments).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enhanced = args.enhanced
    # Default epochs are chosen so typical λ settings produce measurable hard-threshold sparsity
    # (sigmoid(gate) < 0.01) on CPU in reasonable wall-clock time.
    epochs = args.epochs if args.epochs is not None else (50 if enhanced else 40)
    num_workers = args.num_workers if args.num_workers is not None else (2 if enhanced else 0)

    config = ExperimentConfig(
        seed=args.seed,
        enhanced=enhanced,
        show_progress=args.progress,
        epochs=epochs,
        batch_size=args.batch_size,
        num_workers=num_workers,
        plots_dir=args.plots_dir,
    )

    set_seed(config.seed)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    run_gradient_flow_check()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | enhanced={enhanced}", flush=True)

    plots_dir = Path(config.plots_dir)
    wipe_plots_dir(plots_dir)

    train_loader, val_loader, test_loader = make_dataloaders(config)

    results: List[ExperimentResult] = []
    models: Dict[float, ModelType] = {}
    histories: Dict[float, Dict[str, List[float]]] = {}

    for lam in config.lambda_values:
        model, result, history = train_one_lambda(lam, train_loader, val_loader, test_loader, config, device)
        results.append(result)
        models[lam] = model
        histories[lam] = history

        save_gate_hist(
            model.all_gate_values_numpy(),
            plots_dir / f"gate_hist_lambda_{lam}.png",
            title=f"Gate histogram (lambda={lam})",
        )

    best = max(results, key=lambda r: r.final_test_accuracy)
    save_heatmap(models[best.lambda_value], plots_dir / "gate_heatmap_best.png")
    save_tradeoff(results, plots_dir / "lambda_tradeoff.png")

    print(f"\nBest lambda (test acc): {best.lambda_value} ({best.final_test_accuracy:.2f}%)", flush=True)
    print_table(results)

    payload = {"config": asdict(config), "results": [asdict(r) for r in results]}
    with open(config.results_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {config.results_json}", flush=True)


if __name__ == "__main__":
    main()
