# Ballon_Maxxing

A screen-only computer-vision bot for a 60-second browser balloon game.

Google_Max watches the game through the rendered screen, identifies targets, tracks them across frames, understands game-specific mechanics, and sends Windows mouse input to play the game.

The core pipeline is:

```text
Screen
  │
  ▼
MSS Capture
  │
  ▼
Vision
  │
  ├── OpenCV spatial detection
  ├── geometry filtering
  ├── outline / ring detection
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
  ├── four-color sequence
  ├── Super-G / G-symbol
  └── Gemini / multicolor special
  │
  ▼
Fast Execution Policy
  │
  ▼
Balloon Executor
  │
  ▼
Windows SendInput
```

The project is intentionally **screen-only**. It plays from what is visible on screen rather than reading the game's DOM, JavaScript state, network traffic, game seed, or internal engine state.

---

## What It Does

Google_Max currently targets a browser balloon game with:

- Blue balloons
- Red balloons
- Yellow balloons
- Green balloons
- Twin / Golden Duo targets
- A four-color special sequence
- G-symbol / Super-G mechanics
- Gemini / multicolor specials
- White notification objects
- Static UI elements that can resemble balloons

The system is built around the idea that **perception and gameplay logic are different problems**.

Vision determines what appears to be on screen.

Tracking determines which detections belong to the same object over time.

The decision engine determines what those observations mean for the current game state.

The execution layer turns those decisions into mouse actions.

---

# Why I Built This

The first version of Google_Max started with handwritten computer-vision heuristics.

That worked surprisingly well for obvious balloons.

The problem was the edge cases.

Objects such as UI digits and other looped shapes could look remarkably similar to balloons when all the system knew about them was color, area, and geometry. Digits such as `0`, `3`, `6`, `8`, and `9` repeatedly exposed the limits of increasingly complicated heuristic rules.

So the project evolved from:

```text
"Does this shape look like a balloon?"
```

toward:

```text
"Does this candidate actually belong to one of the semantic classes
I care about?"
```

That led to the current hybrid vision system: cheap OpenCV-based candidate generation followed by optional local ML classification.

The handwritten vision is still valuable because it handles the inexpensive spatial filtering before the classifier ever sees a candidate.

---

# Architecture

## Capture

`capture/mss_capture.py`

The game is captured directly from the screen using [MSS](https://python-mss.readthedocs.io/).

Frames flow into the rest of the system as image data plus capture timestamps.

Fullscreen mode can use the primary monitor reported by MSS:

```powershell
python main.py --mode live --game balloons --fullscreen --duration 60
```

Coordinate conversion is kept separate from perception in:

```text
input/coordinates.py
```

---

## Vision

The balloon game's primary vision profile lives in:

```text
game_profiles/balloon_game.py
```

Candidate generation is deliberately multi-stage.

A simplified version of the process is:

```text
HSV color masks
      ↓
morphology
      ↓
connected components
      ↓
area / aspect / fill filtering
      ↓
static UI guards
      ↓
contour geometry
      ↓
circularity / solidity
      ↓
color confidence
      ↓
optional ML classification
```

This lets the cheap operations eliminate most irrelevant pixels before more expensive analysis is performed.

### Geometry

Candidates are evaluated using features including:

- area
- aspect ratio
- fill ratio
- contour circularity
- solidity
- boundary radial consistency
- enclosed-hole characteristics

This matters because color alone is nowhere near enough to distinguish a balloon from arbitrary UI.

### Static UI

The game contains UI elements that can visually resemble targets.

The vision system contains explicit score-counter protection, while the tracker builds additional temporal evidence for objects that remain stationary.

Persistent objects can become:

```text
static_ui_suspect = true
```

and the decision layer can refuse to treat them as actionable balloons.

---

# Special Vision

Several game mechanics require more than ordinary balloon detection.

## Outline Detection

`vision/outline_detector.py`

Some targets have a whitish or translucent boundary.

The detector examines:

- white-pixel ratio around the candidate
- contrast between the candidate and its surrounding region

This produces visual evidence that downstream game logic can use.

---

## Ring Detection

`vision/ring_detector.py`

Some special targets contain a large circular ring.

Ring detection uses:

1. candidate-relative cropping
2. grayscale conversion
3. Gaussian blur
4. Hough-circle detection
5. Canny edge support
6. center validation
7. radius-to-target relationships
8. confidence scoring

The detector is deliberately restricted to plausible candidates rather than running expensive Hough analysis across the entire frame.

---

## G-Symbol Detection

`vision/g_symbol_detector.py`

The G-symbol is identified through its multicolor structure rather than relying on the white center.

The detector looks for the characteristic arrangement of:

- red
- green
- blue
- yellow where applicable

The blue horizontal section provides the click location.

---

# Machine Learning Vision

Google_Max contains a local semantic classifier based on:

**MobileNetV3-Small**

with:

```text
Input:       96 × 96 RGB
Device:      CPU
Framework:   PyTorch / torchvision
Runtime:     batched inference
```

The classifier currently recognizes:

```text
balloon
duo
notification
super_g_ring
ui
```

The runtime pipeline is:

```text
MSS
  ↓
OpenCV candidate generation
  ↓
96×96 candidate crops
  ↓
MobileNetV3-Small
  ↓
semantic class + confidence
  ↓
existing tracker
  ↓
existing decision engine
  ↓
existing execution system
```

The classifier does not replace the rest of the architecture. It plugs into the existing perception pipeline after spatial candidate extraction.

Runtime preprocessing uses OpenCV/NumPy rather than PIL transforms to keep the click-critical path lightweight.

### Runtime configuration

The known working ML configuration is:

```powershell
$env:GOOGLE_MAX_ML_VISION="1"
$env:GOOGLE_MAX_ML_CHECKPOINT="training/artifacts/mobilenet_v3_small.pt"
$env:GOOGLE_MAX_ML_THRESHOLD="0.70"
$env:GOOGLE_MAX_ML_BATCH_SIZE="16"

python main.py --mode live --game balloons --fullscreen --duration 60 --no-gui --verbose-decisions
```

The important controls are:

```text
GOOGLE_MAX_ML_VISION
GOOGLE_MAX_ML_CHECKPOINT
GOOGLE_MAX_ML_THRESHOLD
GOOGLE_MAX_ML_BATCH_SIZE
```

The default checkpoint path is:

```text
training/artifacts/mobilenet_v3_small.pt
```

Training artifacts and raw image data are kept outside the normal source tree.

---

# Tracking

`tracking/tracker.py`

Vision produces observations frame by frame.

Tracking gives those observations temporal identity.

A track stores information such as:

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
- semantic metadata

Matching is spatial and class-aware.

This temporal layer is especially important for distinguishing actual game objects from persistent screen UI.

---

# Game Mechanics

The decision system lives primarily in:

```text
decision/mechanics.py
game_profiles/balloon_types.py
```

The main target types include:

```text
normal_balloon
twin_balloon
special_sequence_balloon
g_symbol
multicolor_special
notification
invalid_white
```

The decision engine receives structured observations and produces actions.

It does **not** directly send mouse input.

That separation makes it possible to inspect what the system thinks should happen before allowing it to actually click.

---

# Normal Balloons

Normal balloons form the primary action lane.

The decision engine filters out objects that should not be clicked, including:

- invalid white objects
- persistent static-UI suspects
- targets that have accumulated excessive click attempts

The last mechanism is particularly useful against persistent false positives: once an object repeatedly survives perception while refusing to disappear, the system stops hammering the same location indefinitely.

---

# Twin / Golden Duo

Twin targets are handled as a coordinated action rather than two unrelated balloons.

The perception and grouping system identifies related same-color targets and associates them with a shared semantic identity.

The decision layer produces a:

```python
TwinAction
```

when the required pair is present.

The execution layer then handles the pair using its own inter-click timing configuration.

The timing constraint is represented explicitly in the action model rather than being hidden inside arbitrary mouse logic.

---

# Four-Color Special Sequence

The special sequence is:

```text
BLUE → RED → YELLOW → GREEN
```

The decision engine owns this state machine.

The system does not trigger the sequence merely because one or two required colors appear.

The required special targets must be observed together before the sequence is armed.

Once active, the decision layer prioritizes the required color in order:

```text
BLUE
  ↓
RED
  ↓
YELLOW
  ↓
GREEN
```

After the sequence completes, the engine returns to normal play.

This state belongs in the decision layer rather than in the vision detector because recognizing a color and knowing **when that color is the next required action** are different problems.

---

# Super-G / G-Symbol

Super-G has its own state rather than being treated as an ordinary balloon.

The vision system can identify a newly appearing cohort containing:

```text
blue + red + yellow + green
```

and use the corresponding ring evidence to validate the special state.

The G-symbol itself is handled independently through its multicolor structure.

This keeps the expensive special-target logic focused on candidates that can plausibly belong to the mechanic.

---

# Gemini / Multicolor Special

Gemini is represented internally as:

```text
TargetType.MULTICOLOR_SPECIAL
```

The current screen-based logic uses observable visual evidence rather than claiming access to hidden game state.

Its detection can use:

- notification presence
- colorful connected components
- plausible size
- aspect ratio
- fill characteristics
- multicolor evidence

The system can therefore reason about the Gemini mechanic from what is actually visible on screen while keeping authoritative game state outside the model.

---

# Execution

The current production execution path is:

```text
behavior/fast_policy.py
```

The fast policy is designed for the click-critical path.

It avoids unnecessary behavioral scheduling overhead while still applying configured timing and endpoint variation.

Different action categories have their own timing ranges, including:

- normal clicks
- G precursors
- Super-G
- multicolor specials
- Twin actions

Normal clicks also use a small configurable endpoint offset rather than repeatedly clicking the exact same pixel.

The result is fast execution without reducing the system to a naive:

```python
click_everything_as_fast_as_possible()
```

loop.

---

# Windows Input

`input/sendinput.py`

Mouse input is sent through the Windows `SendInput` API.

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
Windows SendInput
```

`input/coordinates.py` owns the coordinate conversion.

The application also configures Windows DPI awareness at startup so screen-coordinate mapping is less vulnerable to scaling differences.

---

# Behavioral Model

The repository contains a stochastic behavioral model in:

```text
behavior/model.py
behavior/profile.py
behavior/movement.py
behavior/distributions.py
```

The model can represent:

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
- Twin timing
- Super-G failures
- notification interactions
- early session exit

Behavioral parameters are sampled per session using an entropy-derived seed unless an explicit debug seed is configured.

The important distinction is that this is a **human-oriented behavioral model**, not a claim that the parameters constitute a scientifically validated model of human behavior.

The current live path uses the fast execution policy.

---

# Training the Vision Model

Training infrastructure lives under:

```text
training/
```

The dataset is built from human-labeled screen candidates collected during actual game sessions.

The process is:

```text
game session
    ↓
candidate collection
    ↓
human labeling
    ↓
session-aware dataset split
    ↓
MobileNetV3-Small training
    ↓
evaluation
    ↓
runtime checkpoint
```

## Candidate Collection

Collection mode captures candidate crops without enabling automated clicking.

```powershell
python main.py --mode collect --game balloons --fullscreen --duration 60
```

Candidates are stored under:

```text
training/raw/session_*/
```

The collector also supports recording human clicks so the dataset can contain examples selected from actual interaction.

---

## Human Labeling

The labeling system uses human labels as ground truth.

The available categories include:

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

Detector predictions are not automatically treated as training labels.

---

## Dataset Splitting

The split is **session-aware**.

```powershell
python training/dataset_stats.py training/raw
python training/dataset_split.py training/raw
```

The splitter keeps an entire recording session inside one split, preventing near-identical crops from the same session from appearing across train, validation, and test sets.

The default proportions are:

```text
70% train
15% validation
15% test
```

The assignment is deterministic from the configured seed while attempting to maintain useful class balance.

---

## Training

Install the separate training dependencies:

```powershell
python -m pip install -r training/requirements.txt
```

Then:

```powershell
python training/train.py
```

The training system uses:

- MobileNetV3-Small
- ImageNet pretrained weights
- 96×96 RGB crops
- image augmentation
- class-weighted cross entropy
- validation macro-F1 for model selection
- early stopping

The trained checkpoint is written to:

```text
training/artifacts/mobilenet_v3_small.pt
```

The generated training artifacts and raw dataset are intentionally kept out of the normal source tree.

---

# Runtime Modes

## Live

The normal production mode:

```powershell
python main.py --mode live --game balloons --fullscreen --duration 60
```

For the current ML configuration:

```powershell
$env:GOOGLE_MAX_ML_VISION="1"
$env:GOOGLE_MAX_ML_CHECKPOINT="training/artifacts/mobilenet_v3_small.pt"
$env:GOOGLE_MAX_ML_THRESHOLD="0.70"
$env:GOOGLE_MAX_ML_BATCH_SIZE="16"

python main.py --mode live --game balloons --fullscreen --duration 60 --no-gui --verbose-decisions
```

Live mode enables mouse input.

Hotkeys:

```text
Right Shift → toggle clicking
ESC         → emergency stop
```

A startup delay can be useful when switching to the game window:

```powershell
python main.py --mode live --game balloons --fullscreen --duration 60 --startup-delay 3
```

---

## Dry Run

Dry run exercises perception, tracking, and decision-making without sending automated clicks.

```powershell
python main.py --mode dry-run --game balloons --fullscreen --duration 60 --no-gui --verbose-decisions
```

Useful for inspecting:

- detections
- tracks
- semantic classifications
- special-state transitions
- decision output
- ML predictions
- timing diagnostics

---

## Collection

Collection mode is used to build additional training data:

```powershell
python main.py --mode collect --game balloons --fullscreen --duration 60
```

It is input-free from the automated-clicking side and is intended for gathering candidate crops for later human labeling.

---

# Telemetry and Diagnostics

Telemetry lives under:

```text
telemetry/
```

Diagnostics live under:

```text
diagnostics/
```

The runtime records information about the observable pipeline, including things such as:

- frame timing
- vision timing
- tracking timing
- decision timing
- ML preprocessing/inference timing
- candidate counts
- semantic class distributions
- selected actions
- click timing
- coordinates
- execution outcomes

The diagnostics system deliberately distinguishes between:

**Observed**

Things directly visible from the screen or directly measured by the program.

**Estimated**

Quantities inferred from timestamps and observable events.

**Unobservable**

Authoritative game-engine information that the screen-only interface does not expose.

This distinction matters when analyzing performance. A click being sent is observable. Whether the remote game engine definitively registered that click is not.

---

# Performance

The system has been profiled directly during fullscreen 60-second runs.

The main performance constraint is the vision path rather than screen capture or mouse input.

The architecture therefore tries to keep the expensive work narrow:

```text
cheap spatial filtering
        ↓
small candidate set
        ↓
semantic classification
        ↓
game mechanics
        ↓
click
```

The current ML configuration uses batched CPU inference with:

```text
MobileNetV3-Small
96×96 RGB
batch size: 16
confidence threshold: 0.70
```

The project is optimized around the actual click-critical path rather than maximizing the number of theoretical detections per frame.

---

# Project Structure

```text
Google_Max/
│
├── main.py
│
├── capture/
│   ├── base.py
│   └── mss_capture.py
│
├── vision/
│   ├── color_classifier.py
│   ├── detector.py
│   ├── feedback.py
│   ├── g_symbol_detector.py
│   ├── ml_classifier.py
│   ├── outline_detector.py
│   └── ring_detector.py
│
├── tracking/
│   └── tracker.py
│
├── decision/
│   ├── grouping.py
│   └── mechanics.py
│
├── behavior/
│   ├── action_scheduler.py
│   ├── balloon_executor.py
│   ├── config.py
│   ├── distributions.py
│   ├── execution_policy.py
│   ├── fast_policy.py
│   ├── model.py
│   ├── movement.py
│   └── profile.py
│
├── input/
│   ├── base.py
│   ├── coordinates.py
│   └── sendinput.py
│
├── game_profiles/
│   ├── base.py
│   ├── balloon_game.py
│   ├── balloon_types.py
│   ├── cookie_clicker.py
│   └── ml_balloon_game.py
│
├── capture/
│
├── telemetry/
│   ├── behavioral.py
│   ├── logger.py
│   ├── outcome.py
│   └── timing.py
│
├── diagnostics/
│   ├── decision_console.py
│   └── downstream.py
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
│   └── requirements.txt
│
├── config/
│   ├── defaults.json
│   └── settings.py
│
├── debug/
│   └── visualizer.py
│
└── tests/
```

---

# Testing

The project has a dedicated test suite covering the active perception, tracking, decision, behavior, execution, and training-related components.

The repository is intended to be changed incrementally with the existing tests acting as the regression boundary for the live system.

---

# Cookie Clicker

Cookie Clicker is the original legacy game profile.

It is intentionally still part of the repository because it was the project's first black-box computer-vision experiment and remains functional.

Its implementation is deliberately much simpler:

```text
screen
  ↓
color segmentation
  ↓
cookie geometry
  ↓
center detection
  ↓
click
```

It is useful as the historical starting point for the project and as a minimal example of the `GameProfile` interface.

---

# Design Philosophy

Google_Max is built around a fairly simple idea:

> **Make the machine see the game the way the program is allowed to see it.**

The system does not need to know what the browser thinks is happening.

It needs to turn pixels into increasingly useful representations:

```text
pixels
  ↓
candidates
  ↓
objects
  ↓
tracks
  ↓
game state
  ↓
actions
  ↓
mouse input
```

Each stage has a specific responsibility.

Vision should not own gameplay state.

Tracking should not decide strategy.

The decision engine should not move the mouse.

The executor should not reinterpret what a balloon means.

That separation is what lets the project grow from a collection of OpenCV heuristics into a full perception-and-control system without turning the entire thing into one giant script.

---

# Status

The primary target is the balloon game.

The production path currently consists of:

```text
MSS
→ OpenCV vision
→ optional MobileNetV3-Small semantic classification
→ temporal tracking
→ game-specific decision engine
→ fast execution policy
→ BalloonExecutor
→ Windows SendInput
```

The ML-enabled live path has been exercised end-to-end in 60-second runs.

The repository also retains the collection and training infrastructure used to build and improve the semantic vision system.

---
