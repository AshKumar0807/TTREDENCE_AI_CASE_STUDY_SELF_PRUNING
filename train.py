"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
This script implements a neural network that learns to prune itself during training
using learnable gate parameters and L1 sparsity regularization.

Author: AI Engineering Intern Candidate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import json
import os
import time
from typing import List, Tuple, Dict

# ─────────────────────────────────────────────────────────────
# Part 1: The PrunableLinear Layer
# ─────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate parameters.

    Each weight w_ij has an associated gate score s_ij. The gate is computed
    as gate = sigmoid(s_ij) ∈ (0, 1).  The effective weight used in the
    forward pass is:

        pruned_weight = weight * gate

    When the gate → 0, the weight is effectively removed (pruned) from the
    network without any hard masking — gradients flow cleanly through both
    `weight` and `gate_scores` via standard autograd.

    Args:
        in_features  (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias         (bool): If True, adds a learnable bias. Default: True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight + bias parameters (same init as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Gate scores — same shape as weight; registered as a Parameter so
        # the optimizer updates them alongside the weights.
        # Initialised near 1 (gate ≈ sigmoid(2) ≈ 0.88) so training starts
        # with most connections active and the network can learn to prune.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gates ∈ (0, 1) via sigmoid — differentiable w.r.t. gate_scores
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise multiply: prune weights whose gate → 0
        pruned_weights = self.weight * gates

        # Standard affine transform — gradients propagate to both weight & gate_scores
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from computation graph)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate is below `threshold`."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


# ─────────────────────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 built entirely from PrunableLinear layers.

    Architecture (input 3×32×32 = 3072 features):
        3072 → 1024 → 512 → 256 → 10

    BatchNorm and GELU activations improve training stability and accuracy.
    Dropout is intentionally omitted — sparsity regularisation acts as the
    primary regulariser.
    """

    def __init__(self, input_dim: int = 3 * 32 * 32, num_classes: int = 10):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = F.gelu(self.bn1(self.fc1(x)))
        x = F.gelu(self.bn2(self.fc2(x)))
        x = F.gelu(self.bn3(self.fc3(x)))
        x = self.fc4(x)   # logits — no activation before CrossEntropyLoss

        return x

    def prunable_layers(self) -> List[PrunableLinear]:
        """Return all PrunableLinear layers in order."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 sparsity penalty: sum of all gate values across every PrunableLinear
        layer.  Because gates = sigmoid(scores) > 0, the absolute value is
        redundant — L1 = sum here.

        Minimising this term drives gates toward 0 (pruning connections).
        The L1 norm is preferred over L2 because L1 induces exact zeros
        (corner solutions of the constraint polytope), while L2 only
        shrinks values toward but never exactly to zero.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total = total + gates.sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Global sparsity: fraction of pruned weights across all prunable layers."""
        total_weights = 0
        pruned_weights = 0
        for layer in self.prunable_layers():
            gates = layer.get_gates()
            total_weights += gates.numel()
            pruned_weights += (gates < threshold).sum().item()
        return pruned_weights / total_weights if total_weights > 0 else 0.0

    def total_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ─────────────────────────────────────────────────────────────
# Part 2 + 3: Training & Evaluation
# ─────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2):
    """Download CIFAR-10 and return train / test DataLoaders."""
    # Standard CIFAR-10 normalisation (mean/std per channel)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std =[0.2470, 0.2435, 0.2616]
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def train_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lam: float,
    device: torch.device,
    scheduler=None,
) -> Tuple[float, float, float]:
    """
    Run one training epoch.

    Returns:
        avg_total_loss     : mean of (CrossEntropy + λ·Sparsity) over all batches
        avg_cls_loss       : mean CrossEntropy loss
        avg_sparsity_loss  : mean sparsity loss (unscaled)
    """
    model.train()
    total_loss_sum = cls_loss_sum = sp_loss_sum = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = model.sparsity_loss()
        loss     = cls_loss + lam * sp_loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss_sum += loss.item()
        cls_loss_sum   += cls_loss.item()
        sp_loss_sum    += sp_loss.item()
        n_batches      += 1

    return total_loss_sum / n_batches, cls_loss_sum / n_batches, sp_loss_sum / n_batches


@torch.no_grad()
def evaluate(
    model: SelfPruningNet,
    loader: DataLoader,
    lam: float,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on `loader`.

    Returns:
        accuracy (float) : top-1 accuracy as a percentage
        avg_loss (float) : mean total loss
    """
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = model.sparsity_loss()
        loss     = cls_loss + lam * sp_loss

        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        loss_sum  += loss.item()
        n_batches += 1

    return 100.0 * correct / total, loss_sum / n_batches


def run_experiment(
    lam: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> Dict:
    """
    Train a SelfPruningNet for a given λ value and return results.
    """
    model = SelfPruningNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing over total training steps
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = {
        'train_loss': [], 'train_cls_loss': [], 'train_sp_loss': [],
        'test_acc': [], 'test_loss': [], 'sparsity': [],
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training with λ = {lam}")
        print(f"  Parameters: {model.total_parameters()}")
        print(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_cls, tr_sp = train_epoch(
            model, train_loader, optimizer, lam, device, scheduler
        )
        test_acc, test_loss = evaluate(model, test_loader, lam, device)
        sparsity = model.overall_sparsity()

        history['train_loss'].append(tr_loss)
        history['train_cls_loss'].append(tr_cls)
        history['train_sp_loss'].append(tr_sp)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['sparsity'].append(sparsity)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Loss {tr_loss:.4f} (cls {tr_cls:.4f} + sp {tr_sp:.2f}) | "
                f"Test Acc {test_acc:.2f}% | Sparsity {sparsity*100:.1f}% | "
                f"{elapsed:.0f}s"
            )

    final_acc, _ = evaluate(model, test_loader, lam, device)
    final_sparsity = model.overall_sparsity()

    if verbose:
        print(f"\n  ✓ Final Test Accuracy : {final_acc:.2f}%")
        print(f"  ✓ Final Sparsity      : {final_sparsity*100:.2f}%")

    return {
        'lambda': lam,
        'model': model,
        'final_accuracy': final_acc,
        'final_sparsity': final_sparsity,
        'history': history,
    }


# ─────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────

def plot_gate_distribution(model: SelfPruningNet, lam: float, save_path: str):
    """
    Histogram of final gate values across all PrunableLinear layers.
    A well-pruned network shows a large spike near 0 and a cluster near 1.
    """
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(layer.get_gates().cpu().numpy().ravel())
    all_gates = np.concatenate(all_gates)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_gates, bins=100, color='steelblue', edgecolor='none', alpha=0.85)
    ax.set_xlabel('Gate Value', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'Gate Value Distribution  (λ = {lam})', fontsize=14)
    ax.axvline(x=0.01, color='tomato', linestyle='--', linewidth=1.5, label='Prune threshold (0.01)')
    ax.legend(fontsize=11)

    sparsity = (all_gates < 0.01).mean() * 100
    ax.text(0.65, 0.92, f'Sparsity = {sparsity:.1f}%', transform=ax.transAxes,
            fontsize=12, color='tomato',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Gate distribution plot saved: {save_path}")


def plot_training_curves(results: List[Dict], save_path: str):
    """Plot accuracy and sparsity curves for all λ values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    for i, res in enumerate(results):
        lam  = res['lambda']
        hist = res['history']
        col  = colors[i % len(colors)]
        epochs = range(1, len(hist['test_acc']) + 1)

        axes[0].plot(epochs, hist['test_acc'], color=col, label=f'λ={lam}', linewidth=2)
        axes[1].plot(epochs, [s * 100 for s in hist['sparsity']], color=col,
                     label=f'λ={lam}', linewidth=2)

    axes[0].set_title('Test Accuracy over Training', fontsize=13)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title('Network Sparsity over Training', fontsize=13)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Sparsity (%)')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('Self-Pruning Network — Training Dynamics', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Training curves saved: {save_path}")


def plot_sparsity_vs_accuracy(results: List[Dict], save_path: str):
    """Scatter plot: sparsity level vs test accuracy for different λ."""
    lambdas   = [r['lambda'] for r in results]
    accs      = [r['final_accuracy'] for r in results]
    sparsities = [r['final_sparsity'] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(sparsities, accs, c=lambdas, cmap='viridis', s=150,
                         zorder=5, edgecolors='black', linewidths=0.8)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('λ (sparsity weight)', fontsize=11)

    for lam, sp, acc in zip(lambdas, sparsities, accs):
        ax.annotate(f'λ={lam}', (sp, acc), textcoords='offset points',
                    xytext=(8, 4), fontsize=10)

    ax.set_xlabel('Sparsity Level (%)', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Sparsity vs. Accuracy Trade-off', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Sparsity vs accuracy plot saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    # ── Configuration ──────────────────────────────────────
    EPOCHS      = 30        # Increase to 50-60 for higher accuracy
    BATCH_SIZE  = 128
    LR          = 3e-3
    WEIGHT_DECAY= 1e-4
    LAMBDAS     = [1e-5, 1e-4, 5e-4]   # low / medium / high sparsity pressure
    OUTPUT_DIR  = './outputs'
    # ────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥  Device: {device}")

    print("\n📦  Loading CIFAR-10 ...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE, num_workers=0  # 0 for portability
    )
    print(f"    Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # ── Run experiments for each λ ──────────────────────────
    all_results = []
    for lam in LAMBDAS:
        res = run_experiment(
            lam=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            verbose=True,
        )
        all_results.append(res)

        # Save gate-distribution plot for each λ
        plot_gate_distribution(
            res['model'], lam,
            os.path.join(OUTPUT_DIR, f'gate_dist_lambda_{lam}.png')
        )

    # ── Summary table ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Lambda':<12} {'Test Acc (%)':<18} {'Sparsity Level (%)'}")
    print(f"  {'-'*50}")
    for res in all_results:
        print(
            f"  {res['lambda']:<12} "
            f"{res['final_accuracy']:<18.2f} "
            f"{res['final_sparsity']*100:.2f}"
        )
    print(f"{'='*60}\n")

    # ── Plots ───────────────────────────────────────────────
    plot_training_curves(all_results, os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plot_sparsity_vs_accuracy(all_results, os.path.join(OUTPUT_DIR, 'sparsity_vs_accuracy.png'))

    # ── Save numeric results as JSON ────────────────────────
    summary = [
        {
            'lambda': r['lambda'],
            'final_accuracy': round(r['final_accuracy'], 4),
            'final_sparsity_pct': round(r['final_sparsity'] * 100, 4),
            'history': {k: [round(v, 6) for v in vs]
                        for k, vs in r['history'].items()
                        if k != 'model'},
        }
        for r in all_results
    ]
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Results JSON saved.")

    # ── Save best model checkpoint ──────────────────────────
    best = max(all_results, key=lambda r: r['final_accuracy'])
    torch.save(best['model'].state_dict(),
               os.path.join(OUTPUT_DIR, f'best_model_lambda_{best["lambda"]}.pt'))
    print(f"  Best model (λ={best['lambda']}, acc={best['final_accuracy']:.2f}%) saved.\n")


if __name__ == '__main__':
    main()
