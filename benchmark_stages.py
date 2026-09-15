from __future__ import annotations

import sys
import os
import time
import statistics

import numpy as np
import mss
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import load_settings
from vision.pipeline import VisionPipeline
from tracking.tracker import Tracker
from decision.selector import TargetSelector


def bench_mss_capture(sct, monitor, iterations):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        sct.grab(monitor)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return times


def bench_mss_to_numpy(sct, monitor, iterations):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        raw = sct.grab(monitor)
        arr = np.asarray(raw)[:, :, :3]
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return times


def bench_vision_only(frame, vision, iterations):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        vision.process(frame)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return times


def bench_capture_vision(sct, monitor, vision, iterations):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        raw = sct.grab(monitor)
        arr = np.asarray(raw)[:, :, :3]
        vision.process(arr)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return times


def bench_capture_vision_tracking(sct, monitor, vision, tracker, iterations):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        raw = sct.grab(monitor)
        arr = np.asarray(raw)[:, :, :3]
        dets, _ = vision.process(arr)
        tracker.update(dets)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return times


def bench_capture_vision_tracking_decision(sct, monitor, vision, tracker, selector, iterations):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        raw = sct.grab(monitor)
        arr = np.asarray(raw)[:, :, :3]
        dets, _ = vision.process(arr)
        tracks = tracker.update(dets)
        selector.select(tracks)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return times


def bench_capture_overhead(sct, monitor, iterations):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        sct.grab(monitor)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return times


def bench_numpy_conversion_only(sct, monitor, iterations):
    raw_frames = []
    for _ in range(iterations):
        raw_frames.append(sct.grab(monitor))

    times = []
    for raw in raw_frames:
        t0 = time.perf_counter_ns()
        arr = np.asarray(raw)[:, :, :3]
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return times


def report(label, times):
    arr = np.array(times)
    p50 = np.median(arr)
    p95 = np.percentile(arr, 95)
    fps = 1000.0 / p50 if p50 > 0 else 0
    print(f"  {label:40s}  median={p50:7.2f}ms  p95={p95:7.2f}ms  ~{fps:6.1f} FPS")


def run_resolution(sct, w, h, settings):
    monitor = {"top": 0, "left": 0, "width": w, "height": h}
    vision = VisionPipeline(settings)
    tracker = Tracker(settings)
    selector = TargetSelector(settings)

    WARMUP = 20
    MEASURE = 100

    print(f"\n{'='*70}")
    print(f"  {w}x{h}")
    print(f"{'='*70}")

    for _ in range(WARMUP):
        raw = sct.grab(monitor)
        arr = np.asarray(raw)[:, :, :3]
        vision.process(arr)

    t_capture = bench_mss_capture(sct, monitor, MEASURE)
    t_capture_numpy = bench_mss_to_numpy(sct, monitor, MEASURE)

    raw_for_vision = sct.grab(monitor)
    frame_for_vision = np.asarray(raw_for_vision)[:, :, :3]
    t_vision = bench_vision_only(frame_for_vision, vision, MEASURE)

    t_cap_vis = bench_capture_vision(sct, monitor, vision, MEASURE)
    t_cap_vis_track = bench_capture_vision_tracking(sct, monitor, vision, tracker, MEASURE)
    t_cap_vis_track_dec = bench_capture_vision_tracking_decision(sct, monitor, vision, tracker, selector, MEASURE)

    cap_median = np.median(t_capture)
    capnp_median = np.median(t_capture_numpy)
    vis_median = np.median(t_vision)
    capvis_median = np.median(t_cap_vis)
    captrack_median = np.median(t_cap_vis_track)
    capdec_median = np.median(t_cap_vis_track_dec)

    numpy_overhead = capnp_median - cap_median
    vision_from_capture = capvis_median - capnp_median
    tracking_overhead = captrack_median - capvis_median
    decision_overhead = capdec_median - captrack_median
    total = capdec_median

    print(f"\n  {'Stage':40s}  {'Median (ms)':>12s}  {'P95 (ms)':>12s}  {'FPS':>8s}")
    print(f"  {'-'*40}  {'-'*12}  {'-'*12}  {'-'*8}")

    report("1. MSS capture only", t_capture)
    report("2. MSS + NumPy conversion", t_capture_numpy)
    report("   (NumPy overhead only)", [numpy_overhead])
    report("3. Vision on pre-captured frame", t_vision)
    report("4. Capture + Vision", t_cap_vis)
    report("   (Vision from capture)", [vision_from_capture])
    report("5. Capture + Vision + Tracking", t_cap_vis_track)
    report("   (Tracking overhead)", [tracking_overhead])
    report("6. Capture + Vision + Track + Decision", t_cap_vis_track_dec)
    report("   (Decision overhead)", [decision_overhead])

    print(f"\n  {'TOTAL':40s}  {total:7.2f}ms        ~{1000/total:6.1f} FPS")

    cap_p95 = np.percentile(t_capture, 95)
    vis_p95 = np.percentile(t_vision, 95)
    total_p95 = np.percentile(t_cap_vis_track_dec, 95)
    print(f"\n  P95 breakdown: capture={cap_p95:.2f}ms  vision={vis_p95:.2f}ms  total={total_p95:.2f}ms")


def inspect_mss():
    print("\n" + "="*70)
    print("  MSS INSPECTION")
    print("="*70)

    sct = mss.MSS()
    mon = sct.monitors[1]
    print(f"  Primary monitor: {mon['width']}x{mon['height']}")
    print(f"  Monitors list: {len(sct.monitors)} entries")

    monitor = {"top": 0, "left": 0, "width": 640, "height": 480}

    print("\n  Testing capture timing consistency...")
    times = []
    for _ in range(50):
        t0 = time.perf_counter_ns()
        sct.grab(monitor)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)

    arr = np.array(times)
    print(f"  50 captures at 640x480:")
    print(f"    min={np.min(arr):.2f}ms  median={np.median(arr):.2f}ms  max={np.max(arr):.2f}ms")
    print(f"    std={np.std(arr):.2f}ms  p95={np.percentile(arr, 95):.2f}ms")

    diffs = np.diff(arr)
    large_jumps = np.sum(diffs > 5.0)
    print(f"    Frame-to-frame jumps > 5ms: {large_jumps}/{len(diffs)}")

    print("\n  Testing if timing correlates with frame content...")
    times_light = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        sct.grab(monitor)
        t1 = time.perf_counter_ns()
        times_light.append((t1 - t0) / 1e6)

    big_monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    times_heavy = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        sct.grab(big_monitor)
        t1 = time.perf_counter_ns()
        times_heavy.append((t1 - t0) / 1e6)

    print(f"    640x480 median: {np.median(times_light):.2f}ms")
    print(f"    1920x1080 median: {np.median(times_heavy):.2f}ms")
    print(f"    Ratio: {np.median(times_heavy)/np.median(times_light):.2f}x")

    sct.close()


def main():
    settings = load_settings()

    inspect_mss()

    sct = mss.MSS()

    resolutions = [
        (1920, 1080),
        (1280, 720),
        (800, 600),
        (400, 300),
    ]

    for w, h in resolutions:
        run_resolution(sct, w, h, settings)

    sct.close()

    print(f"\n{'='*70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
