from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_loaders(data_dir: Path, image_size: int, batch_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    train = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    if train.classes != val.classes or train.classes != test.classes:
        raise ValueError(
            f"Class mismatch: train={train.classes}, val={val.classes}, test={test.classes}"
        )

    kwargs = {"batch_size": batch_size, "num_workers": 0}
    return (
        train.classes,
        DataLoader(train, shuffle=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
    )


def make_model(num_classes: int) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the candidate-patch classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("training/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/artifacts"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    classes, train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, args.image_size, args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Classes: {classes}")
    print(f"Train/val/test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_accuracy = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_val:
            best_val = val_accuracy
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": args.image_size,
                },
                args.output_dir / "best_model.pt",
            )

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_accuracy = evaluate(model, test_loader, device)

    (args.output_dir / "classes.json").write_text(
        json.dumps(classes, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"best_val_accuracy={best_val:.4f}")
    print(f"test_accuracy={test_accuracy:.4f}")
    print(f"checkpoint={args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
