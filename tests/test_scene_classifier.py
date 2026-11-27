"""Scene classifier inference: class names come from the model's own class_mapping.json (ISSUE-017).

A fake ONNX session stands in for a model, so no model file or onnxruntime
inference is needed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from runitback.cv import scene_classifier as sc
from runitback.cv.scene_classifier import LEGACY_CLASS_ORDER, SceneClassifier, SceneState, load_class_order

TOOLS = Path(__file__).resolve().parent.parent / "tools"

EIGHT = ["both_airborne", "both_grounded", "both_offstage", "non_gameplay",
         "one_airborne", "one_offstage", "one_on_ledge", "respawning"]


def _write_mapping(folder: Path, names: list[str]) -> Path:
    (folder / "class_mapping.json").write_text(json.dumps({str(i): n for i, n in enumerate(names)}))
    model = folder / "scene_classifier.onnx"
    model.write_bytes(b"not a real model")
    return model


def test_a_model_without_class_mapping_is_the_legacy_deployed_model(tmp_path):
    assert load_class_order(tmp_path / "scene_classifier.onnx") == LEGACY_CLASS_ORDER


def test_class_names_are_read_from_class_mapping(tmp_path):
    assert load_class_order(_write_mapping(tmp_path, EIGHT)) == EIGHT


def test_a_mapping_with_missing_indices_is_rejected(tmp_path):
    (tmp_path / "class_mapping.json").write_text(json.dumps({"0": "both_airborne", "2": "respawning"}))
    with pytest.raises(KeyError):
        load_class_order(tmp_path / "scene_classifier.onnx")


class _FakeSession:
    """Answers with fixed logits; reports n_outputs like an ONNX session."""

    def __init__(self, n_outputs: int, winner: int = 0):
        self.n_outputs, self.winner = n_outputs, winner

    def get_inputs(self):
        return [type("Input", (), {"name": "input"})()]

    def get_outputs(self):
        return [type("Output", (), {"shape": ["batch", self.n_outputs]})()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _outputs, _feeds):
        logits = np.zeros((1, self.n_outputs), dtype=np.float32)
        logits[0, self.winner] = 5.0
        return [logits]


def _classifier(monkeypatch, model: Path, session: _FakeSession) -> SceneClassifier:
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **k: session)
    return SceneClassifier(model_path=model, temporal_smoothing=0)


def test_predictions_are_named_from_the_models_own_classes(tmp_path, monkeypatch):
    # Index 6 is one_on_ledge in this model; the old fixed list called it respawning,
    # and index 7 did not exist in it at all.
    frame = np.zeros((36, 64, 3), dtype=np.uint8)
    model = _write_mapping(tmp_path, EIGHT)
    assert _classifier(monkeypatch, model, _FakeSession(8, winner=6)).classify(frame).state == SceneState.ONE_ON_LEDGE
    result = _classifier(monkeypatch, model, _FakeSession(8, winner=7)).classify(frame)
    assert result.state == SceneState.RESPAWNING and set(result.raw_probabilities) == set(EIGHT)


def test_a_model_whose_outputs_do_not_match_its_class_list_is_disabled(tmp_path, monkeypatch):
    clf = _classifier(monkeypatch, _write_mapping(tmp_path, EIGHT[:2]), _FakeSession(8))
    assert not clf.available
    assert clf.classify(np.zeros((36, 64, 3), dtype=np.uint8)).state == SceneState.UNKNOWN


def test_every_labelled_class_is_a_scene_state():
    spec = importlib.util.spec_from_file_location("label_segments", TOOLS / "label_segments.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["label_segments"] = module
    spec.loader.exec_module(module)
    assert set(module.SCENE_CLASSES) == set(sc.SceneClassifier.CLASS_TO_STATE)


def test_screens_before_and_after_play_are_not_gameplay():
    assert not SceneState.PRE_GAME.is_gameplay
    assert not SceneState.GAME_END.is_gameplay
    assert SceneState.FINISHING_BLOW.is_gameplay
    assert SceneState.RESPAWNING.is_gameplay
