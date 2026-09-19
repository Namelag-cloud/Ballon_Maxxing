from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import cv2
import numpy as np


IMAGE_SIZE = 96
DEFAULT_PADDING_PX = 18
DEFAULT_BATCH_SIZE = 16
DEFAULT_THRESHOLD = 0.70

LABELS = ("balloon", "duo", "notification", "super_g_ring", "ui")


class MLVisionClassifier:
    """Small local image classifier used after cheap spatial candidate extraction."""

    def __init__(
        self,
        checkpoint: str | Path = "training/artifacts/mobilenet_v3_small.pt",
        *,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        padding_px: int = DEFAULT_PADDING_PX,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        import torch
        from training.train_classifier import IMAGE_SIZE as TRAIN_IMAGE_SIZE
        from training.train_classifier import make_model

        if device != "cpu":
            raise ValueError("MLVisionClassifier currently supports CPU only")

        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = max(1, int(batch_size))
        self.padding_px = max(0, int(padding_px))
        self.threshold = float(threshold)

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"ML classifier checkpoint not found: {checkpoint_path}. "
                "Train it first with python -m training.train_classifier."
            )

        checkpoint_data = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.classes = tuple(str(x) for x in checkpoint_data["classes"])
        if not self.classes:
            raise ValueError("ML classifier checkpoint contains no classes")

        self._image_size = int(checkpoint_data.get("image_size", TRAIN_IMAGE_SIZE))
        self._model = make_model(len(self.classes), pretrained=False).to(self._device)
        self._model.load_state_dict(checkpoint_data["state_dict"])
        self._model.eval()

        self._last_stats = {
            "candidate_count": 0,
            "classified_count": 0,
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "total_ms": 0.0,
        }

        # Runtime preprocessing intentionally stays NumPy/OpenCV based.
        # The training pipeline may use torchvision, but PIL transforms add
        # avoidable per-candidate overhead on the click-critical path.
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def last_stats(self) -> dict:
        return dict(self._last_stats)

    def classify_detections(
        self,
        frame_bgr: np.ndarray,
        detections: list[dict],
    ) -> list[dict]:
        """Attach ML predictions to OpenCV candidates without changing game state."""
        if not detections:
            self._last_stats = {
                "candidate_count": 0,
                "classified_count": 0,
                "preprocess_ms": 0.0,
                "inference_ms": 0.0,
                "total_ms": 0.0,
            }
            return []

        total_started = time.perf_counter()
        self._last_stats = {
            "candidate_count": len(detections),
            "classified_count": 0,
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "total_ms": 0.0,
        }
        height, width = frame_bgr.shape[:2]
        preprocess_started = time.perf_counter()
        prepared: list[tuple[dict, tuple[int, int, int, int]]] = []
        tensors = []

        for detection in detections:
            bbox = detection.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x, y, w, h = [int(round(float(v))) for v in bbox[:4]]
            if w <= 0 or h <= 0:
                continue

            x0 = max(0, x - self.padding_px)
            y0 = max(0, y - self.padding_px)
            x1 = min(width, x + w + self.padding_px)
            y1 = min(height, y + h + self.padding_px)
            if x1 <= x0 or y1 <= y0:
                continue

            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            resized = cv2.resize(
                crop,
                (self._image_size, self._image_size),
                interpolation=cv2.INTER_AREA,
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
            array *= (1.0 / 255.0)
            array = (array - self._mean) / self._std
            tensors.append(self._torch.from_numpy(array))
            prepared.append((detection, (x0, y0, x1 - x0, y1 - y0)))

        self._last_stats["preprocess_ms"] = (time.perf_counter() - preprocess_started) * 1000.0
        if not tensors:
            self._last_stats["total_ms"] = (time.perf_counter() - total_started) * 1000.0
            return []

        predictions: list[dict] = []
        inference_started = time.perf_counter()
        with self._torch.inference_mode():
            for start in range(0, len(tensors), self._batch_size):
                batch = self._torch.stack(tensors[start:start + self._batch_size]).to(self._device)
                probabilities = self._torch.softmax(self._model(batch), dim=1)
                confidences, indices = probabilities.max(dim=1)
                for confidence, index in zip(
                    confidences.detach().cpu().tolist(),
                    indices.detach().cpu().tolist(),
                ):
                    predictions.append(
                        {
                            "label": self.classes[int(index)],
                            "confidence": float(confidence),
                        }
                    )
        self._last_stats["inference_ms"] = (time.perf_counter() - inference_started) * 1000.0

        result: list[dict] = []
        for (detection, padded_bbox), prediction in zip(prepared, predictions):
            enriched = dict(detection)
            metadata = dict(enriched.get("metadata") or {})
            metadata.update(
                {
                    "ml_class": prediction["label"],
                    "ml_confidence": prediction["confidence"],
                    "ml_padded_bbox": list(padded_bbox),
                }
            )
            enriched["metadata"] = metadata
            enriched["ml_class"] = prediction["label"]
            enriched["ml_confidence"] = prediction["confidence"]
            result.append(enriched)
        self._last_stats["classified_count"] = len(result)
        self._last_stats["total_ms"] = (time.perf_counter() - total_started) * 1000.0
        return result
