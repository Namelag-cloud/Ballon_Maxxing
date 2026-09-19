# Offline ML classifier experiment

This directory contains an offline experiment for a cheap 96x96 candidate-patch
classifier. It does **not** change the live vision or click path.

## Dataset

The source of truth is the human-labeled data under:

```
training/raw/session_*/labels.json
training/raw/session_*/crops/*.png
```

Current labels are:

- `balloon`
- `duo`
- `super_g_ring`
- `notification`
- `ui`
- `unknown`

Do not auto-label from detector predictions. Human labels remain ground truth.

## 1. Audit and session-aware split

From the repository root:

```powershell
python training/dataset_stats.py training/raw
python training/dataset_split.py training/raw
```

The split script audits class/session composition and writes:

```
training/artifacts/splits/train.jsonl
training/artifacts/splits/val.jsonl
training/artifacts/splits/test.jsonl
training/artifacts/splits/split_summary.json
```

A whole recording session belongs to exactly one split. Individual crops from
the same session are never split across train/val/test, preventing near-identical
frames from leaking between evaluation sets.

The splitter uses a deterministic seed and searches for a class-balanced
session assignment. With very rare classes such as `duo`, the summary makes
the available evaluation support explicit rather than inventing examples.

The generated artifacts and raw images are gitignored.

## 2. Train MobileNetV3-Small

Install the separate training dependencies:

```powershell
python -m pip install -r training/requirements.txt
```

Then:

```powershell
python training/train.py
```

Equivalent direct entry point:

```powershell
python training/train_classifier.py
```

Defaults:

- MobileNetV3-Small
- pretrained ImageNet weights
- 96x96 RGB crops
- restrained translation/scale/brightness/contrast augmentation
- class-weighted cross entropy
- CPU training by default
- validation macro-F1 for model selection
- early stopping

A class with zero human examples is not placed in the output head because it
cannot be learned. Once genuine `unknown` examples are collected, rerunning
the split/training pipeline will include that class automatically.

Outputs:

```
training/artifacts/mobilenet_v3_small.pt
training/artifacts/test_report.json
```

The test report contains:

- accuracy
- macro-F1
- per-class precision
- per-class recall
- per-class F1
- support
- confusion matrix

## 3. CPU batch=1 benchmark

After training:

```powershell
python training/benchmark_classifier.py
```

This measures the trained model's forward-pass latency on CPU at batch size 1
after warmup and reports mean, median, p95, p99, and single-stream FPS.

This is intentionally separate from training so model quality and runtime
cost can be evaluated independently.

## 4. Decide whether it belongs in the live path

Do **not** integrate the classifier before reviewing both:

1. per-class test metrics/confusion matrix; and
2. CPU batch=1 latency.

A good overall accuracy is not sufficient. In particular, `ui -> balloon`
errors represent false-click risk, while `balloon -> ui` errors represent
missed-click risk. Rare `duo` performance must be interpreted using its actual
held-out support.

If the model is useful but too expensive to run on every candidate, a later
integration can restrict it to ambiguous candidates. No such integration is
included in this experiment.


## Experimental live ML vision path

The trained MobileNet checkpoint can be enabled without changing the normal runtime path.

The pipeline is:

    MSS -> OpenCV spatial candidates -> MobileNet semantic class -> existing tracker/decision mechanics -> executor

The classifier consumes the spatial candidate crops and predicts:
balloon, duo, notification, super_g_ring, or ui.

The ML path does not run the legacy G-symbol, Super-G, or golden-duo semantic detectors. Those semantics are supplied by the classifier and then passed to the existing game mechanics. Gemini/multicolor remains a narrow mechanics-only observation because the current dataset has no Gemini class; it does not override ML classifications.

Enable the experimental path in PowerShell:

    $env:GOOGLE_MAX_ML_VISION="1"
    python main.py --mode dry-run --game balloons --fullscreen --duration 60 --no-gui --verbose-decisions

Optional controls:

    $env:GOOGLE_MAX_ML_THRESHOLD="0.70"
    $env:GOOGLE_MAX_ML_BATCH_SIZE="16"
    $env:GOOGLE_MAX_ML_CHECKPOINT="training/artifacts/mobilenet_v3_small.pt"

The ML path is opt-in and the default runtime path is unchanged. Do not enable live input until a dry-run has been compared with the baseline.

The runtime profile records spatial-candidate time, ML candidate count, predicted class distribution, accepted/rejected counts, preprocessing time, inference time, and total ML time. Runtime preprocessing uses OpenCV/NumPy rather than PIL transforms to keep the hot path cheap.
