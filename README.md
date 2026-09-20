# Google_Max

A screen-only computer-vision automation system for the timed balloon game.

`Google_Max` observes the game through the rendered screen, identifies game objects using OpenCV-based vision (with an optional local ML semantic classifier), tracks targets across frames, applies game-specific mechanics, schedules clicks, and sends Windows mouse input through `SendInput`.

The system deliberately does **not** depend on DOM access, browser automation APIs, CDP, network traffic, JavaScript injection, or authoritative game state.

---

## What It Does

The primary target is a 60-second browser balloon game containing:

- Normal colored balloons
  - Blue
  - Red
  - Yellow
  - Green
- Twin / Golden Duo balloons
- A four-color special sequence
- The G-symbol
- Gemini / multicolor special
- White notification objects
- Static UI elements that visually resemble targets

The bot is designed as a perception → tracking → mechanics → behavior → input pipeline rather than as a monolithic "find something and click it" script.

The important architectural boundary is:

```text
Screen
  │
  ▼
MSS Capture
  │
  ▼
Vision / Candidate Generation
  │
  ├── OpenCV color + geometry
  ├── outline / border evidence
  ├── ring detection
  ├── G-symbol detection
  └── optional MobileNet semantic classification
  │
  ▼
Target Detections
  │
  ▼
Temporal Tracking
  │
  ▼
Game Mechanics / Decision Engine
  │
  ├── normal balloons
  ├── Twin / Golden Duo
  ├── four-color special sequence
  ├── Super-G state
  ├── G-symbol lane
  └── Gemini / notification relationship
  │
  ▼
Execution Policy
  │
  ├── Fast execution
  └── experimental humanized policy
  │
  ▼
Balloon Executor / Scheduler
  │
  ▼
Windows SendInput
```

Telemetry and diagnostics run alongside the pipeline rather than controlling game state.

---

# Design Principles

### Screen-only

The bot only knows what can be inferred from pixels and its own history.

It does not receive:

- spawn ticks
- pop ticks
- authoritative hit state
- server state
- game seed
- DOM state
- browser event state

Where a quantity cannot be observed from the current interface, the code intentionally records it as unavailable rather than pretending it knows the answer.

### Separate perception from mechanics

Vision answers:

> "What appears to be on screen?"

The decision layer answers:

> "Given what we can currently see, what should be clicked?"

This distinction is particularly important for the four-color sequence, Twin/Gemini behavior, and Super-G state.

### Cheap tests before expensive tests

The balloon detector progressively filters candidates:

```text
HSV mask
→ morphology
→ connected components
→ area / aspect / fill gates
→ score-UI guard
→ contour geometry
→ circularity / solidity
→ color confidence
→ semantic classification
```

More expensive operations such as Hough-circle ring detection are restricted to candidates that have already passed cheaper tests.

### State belongs downstream

The vision profile does not decide gameplay priority.

For example, detecting all four special-sequence colors does not itself cause clicks in the required order. The decision engine owns that state machine.

---

# Repository Structure

```text
Google_Max/
│
├── main.py
│
├── analysis/
│   ├── behavior_validation.py
│   ├── calibration_plan.py
│   └── replay_validation.py
│
├── behavior/
│   ├── action_scheduler.py
│   ├── balloon_executor.py
│   ├── config.py
│   ├── distributions.py
│   ├── execution_policy.py
│   ├── fast_policy.py
│   ├── humanized_policy.py
│   ├── model.py
│   ├── movement.py
│   ├── profile.py
│   └── scheduler.py
│
├── capture/
│   ├── base.py
│   └── mss_capture.py
│
├── config/
│   ├── defaults.json
│   └── settings.py
│
├── debug/
│   └── visualizer.py
│
├── decision/
│   ├── grouping.py
│   ├── mechanics.py
│   ├── rules.py
│   └── selector.py
│
├── diagnostics/
│   ├── decision_console.py
│   └── downstream.py
│
├── game_profiles/
│   ├── base.py
│   ├── balloon_game.py
│   ├── balloon_types.py
│   ├── cookie_clicker.py
│   └── ml_balloon_game.py
│
├── input/
│   ├── base.py
│   ├── coordinates.py
│   └── sendinput.py
│
├── telemetry/
│   ├── behavioral.py
│   ├── logger.py
│   ├── outcome.py
│   └── timing.py
│
├── tests/
│
├── tracking/
│   └── tracker.py
│
├── training/
│   ├── candidate_collector.py
│   ├── dataset_split.py
│   ├── dataset_stats.py
│   ├── label_candidates.py
│   ├── train.py
│   ├── train_classifier.py
│   ├── benchmark_classifier.py
│   ├── README.md
│   └── dataset/
│
├── vision/
│   ├── color_classifier.py
│   ├── detector.py
│   ├── feedback.py
│   ├── g_symbol_detector.py
│   ├── ml_classifier.py
│   ├── outline_detector.py
│   ├── pipeline.py
│   └── ring_detector.py
│
├── benchmark_behavior.py
├── benchmark_stages.py
└── requirements.txt
```

---

# Runtime Pipeline

## 1. Capture

`capture/mss_capture.py` uses [MSS](https://python-mss.readthedocs.io/) to capture a configured screen region.

Frames are returned as:

```python
(bgr_numpy_array, monotonic_timestamp)
```

The default configuration targets a 1920×1080 region, while `--fullscreen` dynamically uses the primary monitor reported by MSS.

Coordinate conversion is handled separately in `input/coordinates.py`.

---

## 2. Vision

The main balloon profile lives in:

```text
game_profiles/balloon_game.py
```

It performs significantly more than generic color segmentation.

### Color segmentation

The four balloon colors are represented by HSV ranges.

OpenCV connected components provide:

- bounding box
- area
- center
- component geometry

### Geometry filtering

Candidates are rejected using several shape constraints, including:

- minimum/maximum area
- aspect ratio
- fill ratio
- circularity
- solidity
- boundary radial consistency
- enclosed-hole ratio

This is important because arbitrary colored UI elements can satisfy a simple hue threshold.

### White objects

White objects are handled separately.

A white candidate must satisfy geometric constraints and additional halo evidence before becoming a notification detection.

Notifications are explicitly non-actionable.

### Static UI rejection

The known score-counter region receives an explicit geometry guard.

The tracker additionally builds temporal evidence for persistent static objects. A target that remains nearly motionless for enough frames can be marked:

```text
static_ui_suspect = true
```

The decision layer can then refuse to click it.

Repeated click attempts on persistent normal-balloon false positives also create a temporary spatial suppression region.

---

# Special Visual Detectors

## Outline detection

`vision/outline_detector.py`

Some game objects have a whitish/translucent boundary.

The detector measures:

- white-pixel ratio around the candidate
- contrast between the candidate interior and surrounding region

This provides a visual feature rather than directly deciding game semantics.

---

## Ring detection

`vision/ring_detector.py`

The large special balloons use a large circular outline.

Ring detection is intentionally expensive, so it is not run blindly across every candidate.

It uses:

1. candidate-relative crop
2. grayscale conversion
3. Gaussian blur
4. Hough circle detection
5. Canny edge support
6. center-distance validation
7. radius-to-balloon ratio
8. confidence scoring

The ring radius is expressed relative to the detected balloon rather than as an absolute screen size.

---

## G-symbol detection

`vision/g_symbol_detector.py`

The large Google-style G is detected using its multicolor spatial structure.

The detector looks for the arrangement of:

- red
- green
- blue
- optional yellow

rather than relying on the white center or click counter.

The blue horizontal bar is used as the click location because it provides a reliable colored click surface.

---

# Game Mechanics

Game-specific semantic state is represented in:

```text
game_profiles/balloon_types.py
decision/mechanics.py
```

Target types include:

```text
normal_balloon
twin_balloon
special_sequence_balloon
g_symbol
multicolor_special
notification
invalid_white
```

---

## Normal balloons

Normal balloons are independently actionable unless rejected by safety/temporal rules.

The decision engine ignores:

- invalid white objects
- persistent static UI suspects
- normal targets that have already accumulated enough click attempts

The latter prevents repeatedly hammering a persistent false positive forever.

Twin/Gemini targets use separate semantics and are not subject to the normal-balloon suppression lane.

---

# Twin / Golden Duo

Two same-color special balloons can form a Twin/Gemini-style pair.

The perception layer assigns the pair a shared semantic ID.

The decision engine only constructs a `TwinAction` when:

- exactly two relevant twin detections are present
- both share the same semantic pair ID
- both have usable coordinates

The executor then handles the two targets as a coordinated action rather than as unrelated normal balloons.

The timing constraint is represented explicitly as:

```python
max_click_delta_ms
```

The current decision contract uses a 750 ms upper bound, while the execution policy applies its own configured inter-click timing.

---

# Four-Color Special Sequence

The special sequence is:

```text
BLUE → RED → YELLOW → GREEN
```

The decision engine does not arm the sequence from a partial observation.

All four required colors must be observed in the same semantic observation before the one-shot sequence is armed.

Once armed:

```text
normal actions
twins
other primary targets
```

do not consume the primary lane until the required special step is completed.

After:

```text
BLUE
RED
YELLOW
GREEN
```

the state returns to normal play and the engine marks the system as ready for the next Super-G stage.

---

# Super-G

Super-G state is deliberately separated from the ordinary G-symbol state.

The balloon vision profile looks for a newly appearing cohort containing:

```text
blue + red + yellow + green
```

and validates the corresponding large rings.

The expensive ring detector is therefore restricted to the small set of candidates that can plausibly form the four-node cohort.

Once the precursor sequence completes, the next appropriate G-labelled special target is promoted into the independent G execution lane.

---

# Gemini / Multicolor Special

Gemini is represented internally as:

```text
TargetType.MULTICOLOR_SPECIAL
```

The current screen-based detector uses a deliberately conservative source-grounded rule:

- at least two notification detections
- colorful connected component
- plausible size
- plausible aspect ratio
- sufficient multicolor score

The detector returns at most one Gemini candidate.

The decision engine treats notification presence as an observable precondition for executable Gemini behavior.

The system does not claim to know the authoritative engine state behind the notification/Gemini relationship.

---

# Tracking

`tracking/tracker.py` provides temporal identity.

Each `BalloonTrack` stores:

- track ID
- position
- bounding box
- color
- confidence
- first/last seen timestamps
- missing-frame count
- click state
- click-attempt count
- displacement statistics
- stationary-frame count
- maximum bounding-box change
- semantic metadata

Matching is currently spatial and class-aware:

```text
existing track
      │
      ├── same tracking class
      └── within max_distance
             │
             ▼
         matched track
```

Expired tracks are removed after the configured missing-frame budget.

Temporal state is particularly important for distinguishing moving game objects from persistent UI.

---

# Decision Layer

`decision/mechanics.py` owns game state.

The decision engine receives structured observations and produces:

```python
DecisionResult(
    actions=...,
    g_action=...,
    multicolor_specials=...,
    notifications=...,
    mode=...,
    special_step=...,
)
```

This gives the runtime separate lanes for:

- primary actions
- G actions
- Gemini observations
- notifications
- special-sequence state

The decision layer intentionally does not send mouse input.

---

# Execution

The execution layer lives under `behavior/`.

The current runtime uses the fast policy by default.

## Fast execution

`behavior/fast_policy.py`

Fast execution bypasses the long behavioral reaction/movement/click-duration scheduler.

It still applies configured inter-click timing and small endpoint offsets.

Different action categories have different timing ranges, including:

- normal clicks
- G precursors
- Super-G
- multicolor special
- twins

This is why "fast" does not mean "click everything in a zero-delay loop."

It means the behavioral scheduling overhead is removed from the click-critical path.

---

# Behavioral Model

The repository also contains a stochastic behavioral model.

`behavior/model.py` can model:

- reaction time
- perception delay
- Fitts-law movement time
- click duration
- endpoint error
- misses
- delayed responses
- jitter
- trajectory variation
- warmup
- fatigue
- micro-pauses
- idle gaps
- Duo timing
- Super-G failures
- notification clicks
- early session exit

A session receives an entropy-derived seed unless an explicit debug seed is configured.

The sampled human profile is stored in telemetry so a session's modeled parameters can be reconstructed.

### Important

The behavioral model is **not a learned human model**.

`behavior/profile.py` explicitly describes its defaults as provisional model parameters rather than measurements of a human population.

The current `HumanizedExecutionPolicy` interface exists, but its `tick()` and `shutdown()` methods currently raise `NotImplementedError`. The implemented production execution path is the fast policy.

---

# Input

`input/sendinput.py` uses the Windows `SendInput` API.

The coordinate path is:

```text
frame coordinates
      ↓
capture-region coordinates
      ↓
screen coordinates
      ↓
virtual-screen normalization
      ↓
SendInput
```

`input/coordinates.py` contains the conversion logic.

The application also sets Windows DPI awareness at startup so coordinate mapping is less vulnerable to DPI scaling mismatches.

---

# Runtime Modes

`main.py` supports four modes.

## Benchmark

```powershell
python main.py --mode benchmark --game balloons --fullscreen --duration 60 --no-gui
```

No mouse input is sent.

The runtime measures processing and end-to-end pipeline timing.

---

## Dry Run

```powershell
python main.py --mode dry-run --game balloons --fullscreen --duration 60 --no-gui --verbose-decisions
```

Runs the perception/tracking/decision pipeline without clicking.

Useful for validating:

- detections
- tracks
- target classification
- special-state transitions
- decision output
- ML classifications

---

## Live

```powershell
python main.py --mode live --game balloons --fullscreen --duration 60
```

Live mode enables mouse input.

Hotkeys:

```text
Right Shift → toggle clicking
ESC         → emergency stop
```

A startup delay can be used to switch to the game window:

```powershell
python main.py --mode live --game balloons --fullscreen --duration 60 --startup-delay 3
```

Live mode forces `input.enabled=True` even if the loaded configuration has it disabled.

---

## Collection

```powershell
python main.py --mode collect --game balloons --fullscreen --duration 60
```

Collection mode is structurally input-free.

It captures detector candidates for later human labeling.

The collector writes candidate crops under:

```text
training/raw/session_*/
```

It deliberately samples persistent/moved candidates rather than blindly saving every frame.

---

# Candidate Labeling

Human labels are treated as ground truth.

The labeler supports:

```text
B → balloon
D → duo
G → super_g_ring
U → ui
N → notification
X → unknown
S → skip
```

Run:

```powershell
python training/label_candidates.py
```

or point it at a particular session.

The project explicitly avoids automatically treating detector predictions as training labels.

---

# ML Vision

The repository contains an optional local semantic classifier based on:

```text
MobileNetV3-Small
96×96 RGB candidate crops
CPU inference
```

The model classes are:

```text
balloon
duo
notification
super_g_ring
ui
```

The architecture is:

```text
MSS
 ↓
cheap OpenCV spatial candidates
 ↓
96×96 candidate crops
 ↓
MobileNetV3-Small
 ↓
semantic class
 ↓
existing tracker
 ↓
existing game mechanics
 ↓
existing executor
```

The ML path intentionally does not reuse the legacy semantic detectors for:

- G-symbol
- Super-G
- golden-duo semantics

Those semantics are supplied by the classifier in the ML profile.

The ML path is opt-in through:

```powershell
$env:GOOGLE_MAX_ML_VISION="1"
```

Example:

```powershell
$env:GOOGLE_MAX_ML_VISION="1"
python main.py --mode dry-run --game balloons --fullscreen --duration 60 --no-gui --verbose-decisions
```

Optional controls:

```powershell
$env:GOOGLE_MAX_ML_THRESHOLD="0.70"
$env:GOOGLE_MAX_ML_BATCH_SIZE="16"
$env:GOOGLE_MAX_ML_CHECKPOINT="training/artifacts/mobilenet_v3_small.pt"
```

The default non-ML runtime remains unchanged unless the environment variable is enabled.

The checkpoint is intentionally not part of the normal Git repository tree. Training artifacts and raw image data are gitignored.

---

# ML Training Pipeline

Training dependencies are separate from runtime dependencies:

```powershell
python -m pip install -r training/requirements.txt
```

which provides:

```text
torch
torchvision
Pillow
```

## Dataset statistics

```powershell
python training/dataset_stats.py training/raw
```

## Session-aware split

```powershell
python training/dataset_split.py training/raw
```

The splitter keeps entire recording sessions together.

That means crops from one recording cannot leak into train/validation/test simultaneously.

Default proportions are:

```text
70% train
15% validation
15% test
```

The split is deterministic from the configured seed.

## Train

```powershell
python training/train.py
```

Equivalent:

```powershell
python training/train_classifier.py
```

The training pipeline uses:

- MobileNetV3-Small
- ImageNet pretrained weights by default
- 96×96 crops
- restrained augmentation
- class-weighted cross entropy
- CPU training by default
- validation macro-F1 for checkpoint selection
- early stopping

Outputs include:

```text
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

## Benchmark the classifier

```powershell
python training/benchmark_classifier.py
```

This separately measures CPU batch-1 inference latency.

---

# Telemetry

Runtime telemetry is written under:

```text
telemetry_output/
```

The logger records per-frame information such as:

- frame ID
- capture timestamp
- detection timestamp
- decision timestamp
- input timestamp
- processing time
- end-to-end latency
- detection count
- track count
- click count
- action timestamps

Session summaries additionally contain behavioral and outcome information.

---

# Outcome Observability

A major design constraint is that screen-only automation cannot directly observe authoritative game outcomes.

`telemetry/outcome.py` therefore distinguishes:

### Observable

Examples:

- when the bot saw a target
- when an action was planned
- when mouse-down occurred
- click coordinates
- modeled endpoint error
- whether a target remained visible on the next frame

### Estimated

Examples:

- estimated acquisition-to-click ticks
- click-tick modulo
- timing error relative to the bot's own intended timing

### Unobservable

Examples:

- authoritative pop tick
- authoritative hit/miss state
- server/game-engine state
- true game seed

The project deliberately keeps these categories separate.

---

# Downstream Diagnostics

`diagnostics/downstream.py` provides a JSONL execution trace.

It can capture events such as:

```text
tracking_decision
frame_to_executor
executor_result
scheduler_start
scheduler_phase
click_context
click_followup
click_loop_latency
decision_state_transition
frame_timing
click_timing
diagnostic_summary
```

This makes it possible to answer questions like:

```text
Was the target detected?
        ↓
Was it selected?
        ↓
Did it reach the executor?
        ↓
Did the executor schedule it?
        ↓
Did SendInput execute?
        ↓
What was visible immediately afterward?
```

The trace is intended as a diagnostic artifact rather than permanent source data.

---

# Analysis Tools

The `analysis/` directory contains tools for examining behavioral telemetry.

### `behavior_validation.py`

Summarizes:

- reaction intervals
- spatial error
- normalized spatial error
- Duo timing
- session duration
- click-tick modulo
- lag-1 reaction autocorrelation
- intended vs observed timing error

It can compare bot telemetry against a separate reference dataset.

### `replay_validation.py`

Validates the structure of captured interaction streams.

It checks:

- malformed interactions
- non-monotonic ticks
- non-finite coordinates

It does **not** claim to replay or validate the authoritative game engine.

### `calibration_plan.py`

Prints the project's intended calibration workflow for dry runs, live sessions, human reference data, and telemetry analysis.

---

# Debugging

The debug visualizer is provided by:

```text
debug/visualizer.py
```

It can render:

- detections
- bounding boxes
- centers
- colors/confidence
- outlines
- track IDs
- selected actions
- FPS
- current input mode

Disable the window with:

```powershell
--no-gui
```

For compact decision-level output:

```powershell
--verbose-decisions
```

---

# Performance Instrumentation

The vision system contains detailed stage-level timing instrumentation.

The balloon profile records timings for components including:

```text
HSV conversion
color masks
morphology
connected components
component analysis
classification
notification detection
G-symbol detection
ring detection
semantic processing
```

The ring detector separately records:

```text
crop
grayscale
blur
Hough
Canny
scoring
total
```

`benchmark_stages.py` provides lower-level capture/vision/tracking/decision benchmarks.

`benchmark_behavior.py` benchmarks the stochastic behavioral model and includes Fitts-law checks.

---

# Testing

The project uses Python's built-in `unittest` runner.

Run the full suite with:

```powershell
python -m unittest discover -s tests -v
```

The test suite covers multiple layers, including:

- action scheduling
- balloon contracts
- balloon vision
- vision profiling
- decision mechanics
- execution integration
- fast execution
- behavior modeling
- behavioral replay validation
- candidate collection
- candidate labeling
- CLI collection behavior
- color classification
- coordinate conversion
- dataset splitting
- dataset statistics
- decision console
- downstream diagnostics
- G-symbol detection
- ML classifier contracts
- ring detection
- semantic invariants
- startup delay
- tracking

GitHub Actions runs the same `unittest` discovery command on Windows with Python 3.11.

---

# CI

The repository's GitHub Actions workflow:

```text
.github/workflows/tests.yml
```

uses:

```text
Windows latest
Python 3.11
requirements.txt
python -m unittest discover -s tests -v
```

The project is Windows-specific because the live input layer depends on the Windows `SendInput` API.

---

# Configuration

Runtime configuration lives primarily in:

```text
config/defaults.json
config/settings.py
```

Important configurable areas include:

```text
capture
colors
detection
outline
tracking
grouping
rules
input
feedback
debug
behavior
```

The default configuration contains the balloon-game HSV thresholds, geometry limits, tracking parameters, grouping parameters, and behavioral parameters.

Custom configuration files can be supplied with:

```powershell
python main.py --config path\to\config.json
```

CLI arguments override relevant runtime settings.

---

# Dependencies

Core runtime:

```text
mss
numpy
opencv-python
pywin32
```

Install with:

```powershell
python -m pip install -r requirements.txt
```

ML/training dependencies:

```text
torch
torchvision
Pillow
```

Install separately with:

```powershell
python -m pip install -r training/requirements.txt
```

Python 3.11 is the CI target.

---

# Secondary Game Profile

The repository also contains a simple Cookie Clicker profile:

```text
game_profiles/cookie_clicker.py
```

It detects the large cookie using HSV color segmentation and shape filtering, then selects its center.

It is accessed with:

```powershell
python main.py --game cookie
```

The primary development and mechanics work in this repository are centered on the balloon game.

---

# Important Runtime Notes

## `--fast` is not "maximum FPS"

Fast execution means the long behavioral scheduler is bypassed.

The fast policy still intentionally inserts configured inter-click timing and endpoint variation.

## Humanized execution is currently incomplete

The interface exists, but:

```text
HumanizedExecutionPolicy.tick()
HumanizedExecutionPolicy.shutdown()
```

currently raise `NotImplementedError`.

The implemented live balloon execution path is the fast policy.

## ML is optional

The normal balloon profile does not require PyTorch.

The ML path is explicitly enabled through:

```text
GOOGLE_MAX_ML_VISION=1
```

and requires a trained checkpoint.

## Training data is separate from source

Raw labeled crops and generated training artifacts are gitignored.

They should not be treated as ordinary source files.

---

# Current Architecture in One View

```text
                    ┌─────────────────────┐
                    │      Browser/Game   │
                    └──────────┬──────────┘
                               │ pixels
                               ▼
                    ┌─────────────────────┐
                    │     MSS Capture     │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │       Vision Layer        │
                 │                           │
                 │ HSV / morphology          │
                 │ connected components      │
                 │ geometry                  │
                 │ outline detection         │
                 │ ring detection            │
                 │ G-symbol detection        │
                 │ optional MobileNet        │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Target Model    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Tracking       │
                    │                     │
                    │ identity            │
                    │ temporal evidence   │
                    │ click attempts      │
                    │ static UI detection │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │    Balloon Decision       │
                 │                           │
                 │ normal play               │
                 │ Twin / Duo                │
                 │ B→R→Y→G sequence         │
                 │ Super-G                   │
                 │ G lane                    │
                 │ Gemini / notifications   │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │     Execution Layer       │
                 │                           │
                 │ FastExecutionPolicy       │
                 │ BalloonExecutor            │
                 │ schedulers                 │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Windows Input     │
                    │     SendInput       │
                    └─────────────────────┘


          ┌──────────────────────────────────────┐
          │ Telemetry / Diagnostics / Analysis   │
          │                                      │
          │ timing • outcomes • traces • replay  │
          └──────────────────────────────────────┘
```

---

# Project Status

The repository currently represents a fairly complete end-to-end automation stack rather than a single vision prototype.

The major layers are separated and testable:

```text
capture
vision
tracking
decision
behavior
execution
input
telemetry
diagnostics
training
analysis
```

The main remaining distinction to keep in mind when working on the project is that **perception, behavioral modeling, and authoritative game outcomes are three different things**.

The bot can observe pixels.

The behavioral model can generate modeled human-like actions.

Neither gives the bot direct access to the game's internal truth.

That separation is intentional and should be preserved when extending the system.
