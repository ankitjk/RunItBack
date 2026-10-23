#!/usr/bin/env python3
"""Train ConvNeXt-Tiny scene classifier for SSBU gameplay.

Trains a scene classifier to identify 9 gameplay states + non_gameplay.
Uses transfer learning from ImageNet-pretrained ConvNeXt-Tiny.

Usage:
    python tools/train_scene_classifier.py [--data-dir DIR] [--epochs N]

Examples:
    # Train with defaults
    python tools/train_scene_classifier.py

    # Custom training
    python tools/train_scene_classifier.py --epochs 50 --batch-size 64 --lr 5e-5
"""

import argparse
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

# Scene classes (order matters - must match label indices)
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

NUM_CLASSES = len(SCENE_CLASSES)


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
    """Compute inverse frequency class weights for imbalanced data."""
    class_counts = torch.zeros(NUM_CLASSES)

    for _, label in dataset.samples:
        class_counts[label] += 1

    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES  # Normalize

    return weights


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
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "class_names": SCENE_CLASSES,
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
    device: torch.device,
) -> None:
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to: {output_path}")


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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Verify class order matches
    if train_dataset.classes != SCENE_CLASSES:
        print("Warning: Dataset classes don't match expected order")
        print(f"  Expected: {SCENE_CLASSES}")
        print(f"  Got: {train_dataset.classes}")

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
    model = create_model(NUM_CLASSES, pretrained=not args.no_pretrained)
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
        save_checkpoint(model, optimizer, epoch, val_acc, args.output_dir, is_best)

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
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Load best model for export
    best_checkpoint = torch.load(args.output_dir / "checkpoint_best.pt", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Export to ONNX
    export_onnx(model, args.output_dir / "scene_classifier.onnx", device)

    # Save training history
    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    # Save class mapping
    class_mapping = {i: name for i, name in enumerate(SCENE_CLASSES)}
    mapping_path = args.output_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Saved class mapping to: {mapping_path}")


if __name__ == "__main__":
    main()
