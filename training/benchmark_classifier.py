from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from training.train_classifier import IMAGE_SIZE, make_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MobileNetV3-Small CPU batch=1.")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/artifacts/mobilenet_v3_small.pt"))
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = make_model(len(classes), pretrained=False).eval()
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cpu()

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    if args.image:
        tensor = transform(Image.open(args.image).convert("RGB")).unsqueeze(0)
    else:
        tensor = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(tensor)

        timings = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            model(tensor)
            timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    mean = sum(timings) / len(timings)
    report = {
        "device": "cpu",
        "batch_size": 1,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "mean_ms": mean,
        "median_ms": timings[len(timings) // 2],
        "p95_ms": timings[int(len(timings) * 0.95) - 1],
        "p99_ms": timings[int(len(timings) * 0.99) - 1],
        "fps_single_stream": 1000.0 / mean if mean else 0.0,
        "classes": classes,
    }
    output = args.checkpoint.with_name("cpu_benchmark.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
