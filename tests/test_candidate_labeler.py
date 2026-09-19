from __future__ import annotations

import json

import pytest

from training.label_candidates import LabelStore, load_crops


def test_label_store_persists_and_resumes(tmp_path):
    path = tmp_path / "labels.json"
    store = LabelStore(path)
    store.set("one.png", "balloon")
    store.set("two.png", "ui")
    store.set("three.png", "skip")

    resumed = LabelStore(path)
    assert resumed.get("one.png") == "balloon"
    assert resumed.get("two.png") == "ui"
    assert resumed.get("three.png") == "skip"
    assert resumed.counts() == {
        "balloon": 1,
        "ui": 1,
        "notification": 0,
        "unknown": 0,
        "skip": 1,
    }


def test_label_store_rejects_unknown_labels(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"one.png": "banana"}), encoding="utf-8")
    with pytest.raises(ValueError):
        LabelStore(path)


def test_label_store_rejects_setting_unknown_label(tmp_path):
    store = LabelStore(tmp_path / "labels.json")
    with pytest.raises(ValueError):
        store.set("one.png", "banana")


def test_load_crops_returns_supported_images_only(tmp_path):
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    (crop_dir / "b.png").write_bytes(b"")
    (crop_dir / "a.jpg").write_bytes(b"")
    (crop_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    assert [path.name for path in load_crops(tmp_path)] == ["a.jpg", "b.png"]
