# Local balloon/UI classifier training

This directory contains the local training scaffold for a tiny candidate-patch classifier.

## Goal

The runtime vision pipeline should stay latency-first:

```
MSS -> cheap OpenCV candidate detection -> tiny classifier (only when needed) -> decision -> SendInput
```

The classifier is intended to distinguish visually similar candidate patches such as:

- `balloon`
- `ui`
- `notification`
- `unknown`

Do not feed the 1920x1080 screen directly to the model. Training data should be small candidate crops, ideally around 64-96 px per side.

## Dataset layout

Populate these folders with image crops:

```
training/dataset/
  train/
    balloon/
    ui/
    notification/
    unknown/
  val/
    balloon/
    ui/
    notification/
    unknown/
  test/
    balloon/
    ui/
    notification/
    unknown/
```

Keep the first dataset balanced. A useful first pass is roughly 500-1,500 crops per class. More difficult UI examples matter more than raw volume.

The most important negatives are the exact false positives seen in live runs: score digits/glyphs that currently become `normal_balloon` candidates.

## Install training dependencies

From the repository root:

```powershell
python -m pip install -r training/requirements.txt
```

## Train the first baseline

```powershell
python training/train.py --data-dir training/dataset --epochs 8 --batch-size 64
```

The script uses a pretrained MobileNetV3-Small backbone and fine-tunes it for the dataset classes. The resulting checkpoint is written under `training/artifacts/`.

## Iteration loop

1. Run the bot with the current fast OpenCV detector.
2. Save candidate crops from false positives and true balloons.
3. Label them into the four classes.
4. Retrain.
5. Measure validation accuracy and, more importantly, false-positive rejection on a held-out set of UI glyphs.
6. Integrate the classifier only after inference latency is measured on the actual Windows machine.

The first model is an experiment, not a reason to slow the click path. If inference adds meaningful latency, it should only run on ambiguous candidates or in a parallel side path.

## Data policy

Do not commit large generated datasets or trained weights to the repository. Keep the directory structure tracked with `.gitkeep` files and keep local images/checkpoints ignored.

## Expected iteration time

For a few thousand 96x96 crops, the first training run should normally be minutes rather than hours. CPU-only training can take tens of minutes depending on the machine; a CUDA-capable GPU should generally make small experiments much faster. The bigger time cost will be collecting representative false-positive examples.

A useful expectation is 3-10 short train/evaluate iterations before deciding whether this approach is actually better than the hand-built geometry pipeline.


## Collecting real screen candidates

The first dataset should come from normal human gameplay, not synthetic screenshots.

Run the collector from the repository root:

```bash
python main.py --mode collect --game balloons --fullscreen --startup-delay 5 --duration 180
```

Collection mode is **screen-capture only**. It disables mouse input and behavioral execution even if a custom config enables them. During the startup delay, switch to the game window. Then play normally and click whatever balloons you can.

The collector runs the existing screen-only OpenCV detector and saves padded candidate patches under:

```
training/raw/session_YYYYMMDD_HHMMSS/
    crops/
    metadata.jsonl
```

Persistent candidates are sampled rather than saved every frame. A candidate can be sampled again after it moves or changes substantially. The raw session directory is gitignored.

Do one short validation run first (about 60–120 seconds). Inspect the number and variety of crops before collecting the larger dataset. A useful first collection target is roughly **5–8 runs of 2–5 minutes each**, then label and inspect the class balance before collecting more.

Important: the collector saves detector candidates, not automatic ground-truth labels. Human labeling remains the source of truth for `balloon`, `ui`, `notification`, and `unknown`.
