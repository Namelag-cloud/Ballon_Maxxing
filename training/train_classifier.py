from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_SIZE = 96
DEFAULT_SEED = 1337


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, classes: list[str], train: bool = False):
        self.items = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.class_to_index = {name: i for i, name in enumerate(classes)}
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.RandomAffine(
                        degrees=3,
                        translate=(0.04, 0.04),
                        scale=(0.95, 1.05),
                    ),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(item["image"]).convert("RGB")
        return self.transform(image), self.class_to_index[item["label"]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_classes(train_manifest: Path) -> list[str]:
    labels = {
        json.loads(line)["label"]
        for line in train_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    # A zero-example class cannot be learned. Keep the output head tied to
    # observed classes and add unknown only when real unknown labels exist.
    return sorted(labels)


def class_weights(dataset: ManifestDataset, classes: list[str]) -> torch.Tensor:
    counts = np.bincount(
        [item["label"] for item in dataset.items],
        minlength=len(classes),
    ).astype(np.float64)
    total = counts.sum()
    weights = total / (len(classes) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def make_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def metrics_from_predictions(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> tuple[dict, list[list[int]]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    per_class = {}
    for i in range(num_classes):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(num_classes)) - tp
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[i] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[i]),
        }
    macro_f1 = sum(x["f1"] for x in per_class.values()) / max(1, num_classes)
    accuracy = sum(matrix[i][i] for i in range(num_classes)) / max(1, len(y_true))
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}, matrix


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    batches = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += float(criterion(logits, labels))
        batches += 1
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    metrics, matrix = metrics_from_predictions(y_true, y_pred, num_classes)
    metrics["loss"] = total_loss / max(1, batches)
    return metrics, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small on labeled crops.")
    parser.add_argument("--splits", type=Path, default=Path("training/artifacts/splits"))
    parser.add_argument("--output", type=Path, default=Path("training/artifacts/mobilenet_v3_small.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    train_manifest = args.splits / "train.jsonl"
    val_manifest = args.splits / "val.jsonl"
    test_manifest = args.splits / "test.jsonl"
    classes = load_classes(train_manifest)
    if not classes:
        raise ValueError("Training split contains no classes")

    train_ds = ManifestDataset(train_manifest, classes, train=True)
    val_ds = ManifestDataset(val_manifest, classes)
    test_ds = ManifestDataset(test_manifest, classes)
    weights = class_weights(train_ds, classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = make_model(len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -math.inf
    best_epoch = 0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        started = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
            batches += 1

        val_metrics, _ = evaluate(model, val_loader, device, len(classes))
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batches),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": IMAGE_SIZE,
                    "seed": args.seed,
                    "best_epoch": epoch,
                },
                args.output,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, matrix = evaluate(model, test_loader, device, len(classes))
    report = {
        "model": "MobileNetV3-Small",
        "classes": classes,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test": test_metrics,
        "confusion_matrix": matrix,
        "history": history,
    }
    report_path = args.output.with_name("test_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
