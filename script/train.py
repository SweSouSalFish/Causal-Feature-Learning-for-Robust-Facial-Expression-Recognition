"""
train.py — Training script with two clear, self-contained branches:
  - main_erm() : standard ERM training
  - main_cicf(): CICF training (ERM + virtual update on G⁺)

Usage:
    # ERM training
    python train.py --mode erm --epochs 50 --train_batch_size 84 --num_workers 0

    # CICF training
    python train.py --mode cicf --epochs 50 --train_batch_size 84 --g_plus_batch_size 256 --k_per_class 3 --num_workers 0
"""

import argparse
import csv
import datetime
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD

from cicf_utils import (
    build_g_plus_batch,
    build_initial_clusters,
    evaluate,
    run_single_step_checks,
    sample_g_plus_batch,
)
from dataset import create_dataloaders, create_deterministic_train_dataset
from model import CICFModel

# Shared utilities (argument parsing, logging, seeding, I/O)

def parse_args():
    parser = argparse.ArgumentParser(description="Train CICF for facial expression classification.")
    parser.add_argument("--mode", type=str, default="cicf", choices=["erm", "cicf"])
    parser.add_argument("--data_dir", type=str, default="dataset_classified_biased")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--g_plus_batch_size", type=int, default=256)
    parser.add_argument("--k_per_class", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_name", type=str, default="")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_logging(args):
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if args.log_name:
        log_file_name = args.log_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"train_{args.mode}_{timestamp}.txt"

    log_path = Path(args.log_dir) / log_file_name
    log_handle = open(log_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)
    return log_handle, original_stdout, original_stderr, log_path


def restore_logging(log_handle, original_stdout, original_stderr):
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(images, labels, device):
    return (
        images.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


def save_history_csv(path, history_rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "epoch",
        "train_loss",
        "train_acc",
        "valid_loss",
        "valid_acc",
        "test_loss",
        "test_acc",
        "epoch_seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(history_rows)


def save_checkpoint(path, model, optimizer, epoch, best_valid_accuracy, args, class_names):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_valid_accuracy": best_valid_accuracy,
            "args": vars(args),
            "class_names": class_names,
        },
        path,
    )


# ERM branch

def train_one_epoch_erm(model, optimizer, train_loader, device):
    """Standard ERM training loop for one epoch."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    total_batches = len(train_loader)
    print(f"Starting ERM epoch with {total_batches} batches")

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        images, labels = move_batch_to_device(images, labels, device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            print(f"Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.4f}")

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = running_loss / max(total_samples, 1)
    accuracy = running_correct / max(total_samples, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def main_erm(args):
    """Full ERM training routine — loads data, trains, evaluates, saves."""

    # -- Data --
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # -- Model & optimizer --
    model = CICFModel(num_classes=args.num_classes, pretrained=args.pretrained).to(args.device)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # -- Print config --
    print("=" * 72)
    print("Training configuration  [ERM]")
    print(f"device: {args.device}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"num_classes: {args.num_classes}")
    print(f"epochs: {args.epochs}")
    print(f"train_batch_size: {args.train_batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"train_samples: {len(loaders['train_dataset'])}")
    print(f"valid_samples: {len(loaders['valid_dataset'])}")
    print(f"test_samples: {len(loaders['test_dataset'])}")
    print(f"checkpoint_path: {os.path.abspath(args.checkpoint_path)}")
    print("=" * 72)

    # -- Training loop --
    best_valid_accuracy = -1.0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch_erm(
            model=model,
            optimizer=optimizer,
            train_loader=loaders["train"],
            device=args.device,
        )

        valid_metrics = evaluate(model, loaders["valid"], args.device)
        test_metrics = evaluate(model, loaders["test"], args.device)
        epoch_seconds = time.time() - epoch_start

        if valid_metrics["accuracy"] > best_valid_accuracy:
            best_valid_accuracy = valid_metrics["accuracy"]
            save_checkpoint(
                path=args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_valid_accuracy=best_valid_accuracy,
                args=args,
                class_names=loaders["train_dataset"].classes,
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "valid_loss": valid_metrics["loss"],
                "valid_acc": valid_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["accuracy"],
                "epoch_seconds": epoch_seconds,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} valid_acc={valid_metrics['accuracy']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} | "
            f"epoch_seconds={epoch_seconds:.2f}"
        )

    # -- Save history --
    history_path = os.path.join(
        Path(args.checkpoint_path).parent,
        f"{args.mode}_history.csv",
    )
    save_history_csv(history_path, history_rows)

    print(f"Best valid accuracy: {best_valid_accuracy:.4f}")
    print(f"Best checkpoint saved to: {os.path.abspath(args.checkpoint_path)}")
    print(f"History saved to: {os.path.abspath(history_path)}")

# CICF branch

def ensure_aligned_samples(train_dataset, deterministic_train_dataset):
    """Verify that the two datasets contain exactly the same samples in order."""
    train_samples = [sample[0] for sample in train_dataset.samples]
    deterministic_samples = [sample[0] for sample in deterministic_train_dataset.samples]
    if train_samples != deterministic_samples:
        raise RuntimeError("Train and deterministic datasets are not aligned by sample order.")


def build_theta_plus(model, g_plus_loss, alpha):
    """Virtual update: θ⁺ = θ_f − α ∇ L(G⁺)."""
    f_named_params = dict(model.f.named_parameters())
    gradients = torch.autograd.grad(
        g_plus_loss,
        tuple(f_named_params.values()),
        create_graph=True,
    )
    theta_plus = {
        name: parameter - alpha * gradient
        for (name, parameter), gradient in zip(f_named_params.items(), gradients)
    }
    return theta_plus


def train_one_epoch_cicf(
    model,
    optimizer,
    train_loader,
    deterministic_train_dataset,
    cluster_state,
    device,
    alpha,
    g_plus_batch_size,
    run_assertions=False,
):
    """One epoch of CICF training (virtual update on G⁺, real update on training batch)."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    assertions_ran = False

    total_batches = len(train_loader)
    print(f"Starting CICF epoch with {total_batches} batches")

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        # 1. Sample G⁺ batch from clusters
        g_plus_indices = sample_g_plus_batch(cluster_state, total_batch_size=g_plus_batch_size)
        g_plus_images, g_plus_labels = build_g_plus_batch(
            deterministic_train_dataset,
            g_plus_indices,
            device=device,
        )

        # 2. Move training batch to device
        images, labels = move_batch_to_device(images, labels, device)

        # 3. Compute virtual update θ⁺ from G⁺ loss
        g_plus_logits = model(g_plus_images)
        g_plus_loss = F.cross_entropy(g_plus_logits, g_plus_labels)
        theta_plus = build_theta_plus(model, g_plus_loss, alpha)

        real_f_param_before = next(model.f.parameters()).detach().clone()

        # 4. Forward with virtual θ⁺ on the real training batch
        logits = model.forward_with_f_params(images, theta_plus)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        # 5. Assertions
        if run_assertions and not assertions_ran:
            h_has_grad = any(
                parameter.grad is not None
                for parameter in model.h.parameters()
                if parameter.requires_grad
            )
            if not h_has_grad:
                raise RuntimeError("No gradient reached h before optimizer.step().")

        optimizer.step()

        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            print(f"Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.4f}")

        if run_assertions and not assertions_ran:
            run_single_step_checks(
                model=model,
                theta_plus=theta_plus,
                first_f_param_before=real_f_param_before,
            )
            assertions_ran = True

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = running_loss / max(total_samples, 1)
    accuracy = running_correct / max(total_samples, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def main_cicf(args):
    """Full CICF training routine — loads data, builds clusters, trains, evaluates, saves."""

    # -- Data --
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    print(f"[{datetime.datetime.now().isoformat()}] Loading deterministic train dataset...")
    deterministic_train_dataset = create_deterministic_train_dataset(
        args.data_dir,
        image_size=args.image_size,
    )
    print(f"[{datetime.datetime.now().isoformat()}] Checking sample alignment...")
    ensure_aligned_samples(loaders["train_dataset"], deterministic_train_dataset)
    print(f"[{datetime.datetime.now().isoformat()}] Sample alignment complete")

    # -- Model & optimizer --
    model = CICFModel(num_classes=args.num_classes, pretrained=args.pretrained).to(args.device)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # -- Build initial clusters (KMeans on extracted features) --
    print(f"[{datetime.datetime.now().isoformat()}] Starting initial cluster construction (KMeans)...")
    cluster_state = build_initial_clusters(
        model=model,
        dataset=deterministic_train_dataset,
        num_clusters_per_class=args.k_per_class,
        batch_size=args.g_plus_batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(f"[{datetime.datetime.now().isoformat()}] Cluster construction complete")

    # -- Print config --
    print("=" * 72)
    print("Training configuration  [CICF]")
    print(f"device: {args.device}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"num_classes: {args.num_classes}")
    print(f"epochs: {args.epochs}")
    print(f"train_batch_size: {args.train_batch_size}")
    print(f"g_plus_batch_size: {args.g_plus_batch_size}")
    print(f"k_per_class: {args.k_per_class}")
    print(f"alpha: {args.alpha}")
    print(f"num_workers: {args.num_workers}")
    print(f"train_samples: {len(loaders['train_dataset'])}")
    print(f"valid_samples: {len(loaders['valid_dataset'])}")
    print(f"test_samples: {len(loaders['test_dataset'])}")
    print(f"checkpoint_path: {os.path.abspath(args.checkpoint_path)}")
    print("=" * 72)

    # -- Training loop --
    best_valid_accuracy = -1.0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch_cicf(
            model=model,
            optimizer=optimizer,
            train_loader=loaders["train"],
            deterministic_train_dataset=deterministic_train_dataset,
            cluster_state=cluster_state,
            device=args.device,
            alpha=args.alpha,
            g_plus_batch_size=args.g_plus_batch_size,
            run_assertions=(epoch == 1),
        )

        valid_metrics = evaluate(model, loaders["valid"], args.device)
        test_metrics = evaluate(model, loaders["test"], args.device)
        epoch_seconds = time.time() - epoch_start

        if valid_metrics["accuracy"] > best_valid_accuracy:
            best_valid_accuracy = valid_metrics["accuracy"]
            save_checkpoint(
                path=args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_valid_accuracy=best_valid_accuracy,
                args=args,
                class_names=loaders["train_dataset"].classes,
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "valid_loss": valid_metrics["loss"],
                "valid_acc": valid_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["accuracy"],
                "epoch_seconds": epoch_seconds,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} valid_acc={valid_metrics['accuracy']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} | "
            f"epoch_seconds={epoch_seconds:.2f}"
        )

    # -- Save history --
    history_path = os.path.join(
        Path(args.checkpoint_path).parent,
        f"{args.mode}_history.csv",
    )
    save_history_csv(history_path, history_rows)

    print(f"Best valid accuracy: {best_valid_accuracy:.4f}")
    print(f"Best checkpoint saved to: {os.path.abspath(args.checkpoint_path)}")
    print(f"History saved to: {os.path.abspath(history_path)}")

# Entry point

def main():
    args = parse_args()
    print(f"[{datetime.datetime.now().isoformat()}] main() started, setting up logging...")
    log_handle, original_stdout, original_stderr, log_path = setup_run_logging(args)
    print(f"[{datetime.datetime.now().isoformat()}] logging setup complete, log_path={log_path}")
    set_seed(args.seed)

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but is not available.")
        args.device = device

        print(f"log_path: {log_path.resolve()}")

        if args.mode == "erm":
            main_erm(args)
        elif args.mode == "cicf":
            main_cicf(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    finally:
        restore_logging(log_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
