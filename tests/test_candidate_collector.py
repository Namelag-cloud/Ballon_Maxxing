from __future__ import annotations

import json

import cv2
import numpy as np

from training.candidate_collector import CandidateCollector


def test_collect_saves_crop_and_metadata(tmp_path):
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[20:40, 30:50] = (0, 0, 255)
    detection = {
        "bbox": (30, 20, 20, 20),
        "center": (40, 30),
        "color_name": "red",
        "target_type": "normal_balloon",
        "confidence": 0.95,
    }

    collector = CandidateCollector(tmp_path, session_name="s1", padding_px=5)
    try:
        assert collector.collect_frame(frame, [detection], frame_id=1, capture_timestamp=1.5) == 1
    finally:
        collector.close()

    crops = list((tmp_path / "s1" / "crops").glob("*.png"))
    assert len(crops) == 1
    image = cv2.imread(str(crops[0]))
    assert image is not None
    assert image.shape[:2] == (30, 30)

    lines = (tmp_path / "s1" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    metadata = json.loads(lines[0])
    assert metadata["bbox"] == [30, 20, 20, 20]
    assert metadata["color"] == "red"
    assert metadata["target_type"] == "normal_balloon"


def test_collect_deduplicates_stationary_candidate(tmp_path):
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    detection = {
        "bbox": (30, 20, 20, 20),
        "center": (40, 30),
        "color_name": "blue",
        "target_type": "normal_balloon",
    }

    collector = CandidateCollector(tmp_path, session_name="s1", min_repeat_frames=20)
    try:
        assert collector.collect_frame(frame, [detection], frame_id=1, capture_timestamp=1.0) == 1
        assert collector.collect_frame(frame, [detection], frame_id=2, capture_timestamp=2.0) == 0
        assert collector.collect_frame(frame, [detection], frame_id=25, capture_timestamp=25.0) == 0
        moved = dict(detection, bbox=(50, 20, 20, 20), center=(60, 30))
        assert collector.collect_frame(frame, [moved], frame_id=26, capture_timestamp=26.0) == 1
    finally:
        collector.close()


def test_collect_handles_edge_clipped_crop(tmp_path):
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    detection = {
        "bbox": (0, 0, 8, 10),
        "color_name": "green",
        "target_type": "normal_balloon",
    }
    collector = CandidateCollector(tmp_path, session_name="s1", padding_px=10)
    try:
        assert collector.collect_frame(frame, [detection], frame_id=1, capture_timestamp=1.0) == 1
    finally:
        collector.close()
    image = cv2.imread(str(next((tmp_path / "s1" / "crops").glob("*.png"))))
    assert image.shape[:2] == (20, 18)


def test_collect_click_saves_every_action_without_dedup(tmp_path):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    collector = CandidateCollector(tmp_path, session_name="s1", click_crop_size_px=96)
    try:
        assert collector.collect_click(
            frame,
            screen_x=80,
            screen_y=60,
            region_left=0,
            region_top=0,
            frame_id=1,
        )
        assert collector.collect_click(
            frame,
            screen_x=80,
            screen_y=60,
            region_left=0,
            region_top=0,
            frame_id=2,
        )
    finally:
        collector.close()

    crops = list((tmp_path / "s1" / "crops").glob("click_*.png"))
    assert len(crops) == 2
    assert all(cv2.imread(str(path)) is not None for path in crops)
    assert not (tmp_path / "s1" / "metadata.jsonl").read_text(encoding="utf-8").strip()


def test_collect_click_handles_screen_region_offset_and_edges(tmp_path):
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    collector = CandidateCollector(tmp_path, session_name="s1", click_crop_size_px=40)
    try:
        assert collector.collect_click(
            frame,
            screen_x=105,
            screen_y=205,
            region_left=100,
            region_top=200,
            frame_id=7,
        )
    finally:
        collector.close()

    path = next((tmp_path / "s1" / "crops").glob("click_*.png"))
    image = cv2.imread(str(path))
    assert image is not None
    assert image.shape[:2] == (40, 40)
