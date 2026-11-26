#!/usr/bin/env python3
"""Train ConvNeXt-Tiny scene classifier for SSBU gameplay.

Fine-tunes ImageNet-pretrained ConvNeXt-Tiny on the scene classes defined in
docs/SCENE_CLASSES.md. Only classes with enough training frames are trained
(--min-class-frames); the held-out test game is scored once at the end.

Usage:
    python tools/train_scene_classifier.py [--data-dir DIR] [--epochs N]

Examples:
    # Train with defaults
    python tools/train_scene_classifier.py

    # Custom training
    python tools/train_scene_classifier.py --epochs 50 --batch-size 64 --lr 5e-5
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

# Every scene class the labeler offers. The model's output order is not this
# list: it is the class folders present in the training split, alphabetically
# (ImageFolder), saved to class_mapping.json.
SCENE_CLASSES = [
    "non_gameplay",
    "both_grounded",
    "one_airborne",
    "both_airborne",
    "one_offstage",
    "both_offstage",
    "one_on_ledge",
    "both_on_ledge",
    "one_knocked_down",
    "both_knocked_down",
    "respawning",
]


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create ConvNeXt-Tiny model with custom classifier head."""
    if pretrained:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
    else:
        model = convnext_tiny(weights=None)

    # Replace classifier head
    # ConvNeXt-Tiny: classifier is Sequential(LayerNorm, Flatten, Linear)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model


def get_transforms(train: bool = True) -> transforms.Compose:
    """Get data transforms for training or validation."""
    # ImageNet normalization (used by ConvNeXt)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),  # SSBU is symmetric
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Compute inverse frequency class weights for imbalanced data.

    One weight per class in the dataset, matching the model's output count.
    Inverse frequency is extreme for a class with a handful of frames (1 frame
    vs 224 is a 224x weight), so main() first drops classes below
    --min-class-frames: with a 1-frame class kept, the model learned to predict
    that class for most frames.
    """
    num_classes = len(dataset.classes)
    class_counts = torch.zeros(num_classes)

    for _, label in dataset.samples:
        class_counts[label] += 1

    # Inverse frequency weighting
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes  # Normalize

    return weights


def reindex_by_name(dataset: datasets.ImageFolder, class_names: list[str]) -> dict[str, int]:
    """Renumber a val/test dataset's labels to match the training class order.

    ImageFolder numbers the class folders present in *its own* directory
    alphabetically. When a split lacks a class the training split has (common
    for rare classes once splits are made by game), its numbers shift and every
    later class would be scored against the wrong model output.

    Frames of a class absent from training are dropped. Returns {class: frames dropped}.
    """
    train_index = {name: i for i, name in enumerate(class_names)}
    dropped: dict[str, int] = {}
    samples = []
    for path, idx in dataset.samples:
        name = dataset.classes[idx]
        if name in train_index:
            samples.append((path, train_index[name]))
        else:
            dropped[name] = dropped.get(name, 0) + 1
    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [t for _, t in samples]
    dataset.classes = list(class_names)
    dataset.class_to_idx = dict(train_index)
    return dropped


def evaluate_test_set(
    model: nn.Module,
    test_dir: Path,
    class_names: list[str],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict | None:
    """Score the model once on the held-out test split.

    Returns overall accuracy, per-class recall with frame counts, macro-average
    recall, and the confusion matrix - or None if there is no test split.
    Test frames of a class the model never trained on are counted and skipped.
    """
    if not test_dir.exists() or not any(test_dir.rglob("*.jpg")):
        return None

    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))
    unseen = reindex_by_name(test_dataset, class_names)

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    n = len(class_names)
    confusion = torch.zeros(n, n, dtype=torch.long)  # rows: true class, columns: predicted

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            predicted = model(inputs.to(device)).argmax(1).cpu()
            for t, p in zip(labels.tolist(), predicted.tolist(), strict=True):
                confusion[t, p] += 1

    support = confusion.sum(1)
    correct = confusion.diag()
    per_class = {
        name: {"frames": int(support[i]), "correct": int(correct[i]), "recall": float(correct[i] / support[i])}
        for i, name in enumerate(class_names)
        if support[i] > 0
    }
    total = int(support.sum())
    return {
        "frames": total,
        "accuracy": float(correct.sum() / total) if total else 0.0,
        "macro_recall": sum(c["recall"] for c in per_class.values()) / len(per_class) if per_class else 0.0,
        "per_class": per_class,
        "classes_not_in_test": [name for i, name in enumerate(class_names) if support[i] == 0],
        "unseen_in_training": unseen,
        "class_names": class_names,
        "confusion_matrix": confusion.tolist(),
    }


def print_test_metrics(metrics: dict) -> None:
    """Print the test-set report: per-class recall, then the headline numbers."""
    print("\n" + "=" * 60)
    print("TEST SET (held out; evaluated once, after checkpoint selection)")
    print("=" * 60)
    print(f"  {'class':<20} {'frames':>7} {'correct':>8} {'recall':>7}")
    for name, c in metrics["per_class"].items():
        print(f"  {name:<20} {c['frames']:>7} {c['correct']:>8} {c['recall']:>7.1%}")
    if metrics["classes_not_in_test"]:
        print(f"  No test frames: {', '.join(metrics['classes_not_in_test'])}")
    if metrics["unseen_in_training"]:
        print(f"  Skipped (class never seen in training): {metrics['unseen_in_training']}")
    print(f"\n  Accuracy:     {metrics['accuracy']:.1%} ({metrics['frames']} frames)")
    print(f"  Macro recall: {metrics['macro_recall']:.1%} (mean of per-class recall)")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_acc: float,
    output_dir: Path,
    class_names: list[str],
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "class_names": class_names,
    }

    # Save latest
    torch.save(checkpoint, output_dir / "checkpoint_latest.pt")

    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / "checkpoint_best.pt")
        print(f"  Saved best model (val_acc={val_acc:.4f})")


def export_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: int = 224,
) -> None:
    """Export the model to ONNX, accepting any batch size, and check the result.

    With an example batch of 1, the default exporter in recent PyTorch (seen
    with 2.13) fixes the batch size at 1 even when `dynamic_axes` marks it as
    variable: the file loads but fails on any larger batch (ISSUE-020). So the
    example batch is 2 and the dimension is declared with `dynamic_shapes`.
    The older TorchScript exporter (`dynamo=False`) also works, but is deprecated.

    The export runs on a CPU copy of the model: exported from Apple's MPS
    device, the batch size came out fixed at 2 instead (PyTorch 2.13).
    """
    model = copy.deepcopy(model).cpu().eval()
    example = torch.randn(2, 3, input_size, input_size)

    # An earlier export may have left its weights in a separate file next to the
    # model. This export writes a single file, so remove the stale one.
    stale_weights = output_path.with_name(output_path.name + ".data")
    if stale_weights.exists():
        stale_weights.unlink()

    torch.onnx.export(
        model,
        (example,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes=({0: torch.export.Dim("batch_size", min=1)},),
        external_data=False,
        dynamo=True,
    )
    check_onnx_export(model, output_path, input_size)
    print(f"Exported ONNX model to: {output_path}")


def check_onnx_export(
    model: nn.Module,
    onnx_path: Path,
    input_size: int = 224,
    batch_size: int = 3,
    tolerance: float = 1e-3,
) -> None:
    """Run a batch through the ONNX file and PyTorch; raise if they disagree."""
    import numpy as np
    import onnxruntime as ort

    batch = torch.randn(batch_size, 3, input_size, input_size)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {session.get_inputs()[0].name: batch.numpy()})[0]
    with torch.no_grad():
        torch_out = model(batch).numpy()
    if onnx_out.shape != torch_out.shape:
        raise RuntimeError(f"ONNX export: output shape {onnx_out.shape}, expected {torch_out.shape}")
    diff = float(np.abs(onnx_out - torch_out).max())
    if diff > tolerance:
        raise RuntimeError(f"ONNX export: outputs differ from PyTorch by {diff:.2e} (limit {tolerance:.0e})")
    print(f"  ONNX check: batch of {batch_size} matches PyTorch (max difference {diff:.1e})")


def main():
    parser = argparse.ArgumentParser(description="Train scene classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/scene_classifier/splits"),
        help="Directory containing train/val/test splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models"),
        help="Output directory for checkpoints and model",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--early-stopping", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pretrained weights")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--min-class-frames",
        type=int,
        default=10,
        help="Leave out classes with fewer training frames than this - too few to learn, "
        "and their inverse-frequency weight swamps the loss",
    )
    args = parser.parse_args()

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"

    if not train_dir.exists():
        print(f"Error: Training data not found at {train_dir}")
        print("Run extract_labeled_frames.py first to create the dataset.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))

    manifest_path = args.data_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is None or manifest.get("split_by") != "game":
        print("WARNING: not a game-level split - validation/test scores will be inflated by")
        print("         near-duplicate frames shared across splits. Re-split with extract_labeled_frames.py.")
    else:
        print(f"Game split - test game held out: {manifest['test_game']}")

    # The model's output i is class_names[i]: the class folders in train/ with
    # enough frames, alphabetically (ImageFolder's order). Everything below,
    # including class_mapping.json, uses this order.
    train_counts: dict[str, int] = {}
    for _, idx in train_dataset.samples:
        name = train_dataset.classes[idx]
        train_counts[name] = train_counts.get(name, 0) + 1
    class_names = [c for c in train_dataset.classes if train_counts.get(c, 0) >= args.min_class_frames]
    too_few = {c: n for c, n in train_counts.items() if n < args.min_class_frames}
    if too_few:
        print(f"Left out (fewer than {args.min_class_frames} training frames): {too_few}")
        reindex_by_name(train_dataset, class_names)
    num_classes = len(class_names)
    val_classes = list(val_dataset.classes)
    dropped = reindex_by_name(val_dataset, class_names)
    if dropped:
        print(f"Validation frames dropped (class not in training): {dropped}")
    no_val = [c for c in class_names if c not in val_classes]
    if no_val:
        print(f"No validation frames for: {no_val}")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes ({num_classes}, model output order): {class_names}")
    missing = [c for c in SCENE_CLASSES if c not in class_names and c not in too_few]
    if missing:
        print(f"Not trained (no training frames): {missing}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Model
    model = create_model(num_classes, pretrained=not args.no_pretrained)
    model = model.to(device)

    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train: {train_loss:.4f} / {train_acc:.4f} | "
            f"Val: {val_loss:.4f} / {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{epoch_time:.1f}s"
        )

        # Save checkpoint
        is_best = val_acc > best_val_acc
        save_checkpoint(model, optimizer, epoch, val_acc, args.output_dir, class_names, is_best)

        if is_best:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= args.early_stopping:
            print(f"\nEarly stopping after {epochs_without_improvement} epochs without improvement")
            break

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {total_time / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f} (used to pick the checkpoint - not a test score)")

    # Load best model for export
    best_checkpoint = torch.load(args.output_dir / "checkpoint_best.pt", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Score the chosen checkpoint once on the held-out test split
    test_metrics = evaluate_test_set(
        model, args.data_dir / "test", class_names, device, args.batch_size, args.workers
    )
    if test_metrics is None:
        print("\nNo test split found - test accuracy not measured.")
    else:
        print_test_metrics(test_metrics)
        test_metrics["best_epoch"] = best_checkpoint["epoch"] + 1
        test_metrics["best_val_acc"] = best_val_acc
        test_metrics["left_out_too_few_frames"] = too_few
        test_metrics["split"] = manifest
        metrics_path = args.output_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Saved test metrics to: {metrics_path}")

    # Export to ONNX
    export_onnx(model, args.output_dir / "scene_classifier.onnx")

    # Save training history
    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    # Save class mapping
    class_mapping = {i: name for i, name in enumerate(class_names)}
    mapping_path = args.output_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Saved class mapping to: {mapping_path}")


if __name__ == "__main__":
    main()
