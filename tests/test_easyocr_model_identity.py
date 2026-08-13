"""XR-02 regressions for exact, fail-closed EasyOCR model identity."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cognitive.easyocr_identity as identity_module  # noqa: E402
import cognitive.text_reader as text_reader  # noqa: E402
from cognitive.easyocr_identity import (  # noqa: E402
    EasyOCRModelIntegrityError,
    load_easyocr_model_identity,
    verify_easyocr_models,
)


DETECTOR_SHA256 = (
    "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17"
)
RECOGNIZER_SHA256 = (
    "e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8"
)


@pytest.fixture(autouse=True)
def _reset_reader_state():
    text_reader._READER = None
    text_reader._READER_FAILED = False
    yield
    text_reader._READER = None
    text_reader._READER_FAILED = False


def _synthetic_contract(tmp_path: Path) -> tuple[Path, Path, dict]:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    identity = copy.deepcopy(load_easyocr_model_identity())
    payloads = {
        "detector": b"detector-model-bytes",
        "recognizer": b"recognizer-model-bytes",
    }
    for role, payload in payloads.items():
        artifact = identity["artifacts"][role]
        (model_directory / artifact["filename"]).write_bytes(payload)
        artifact["bytes"] = len(payload)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    return identity_path, model_directory, identity


def test_tracked_identity_pins_both_score_driving_models_and_reader_policy():
    identity = load_easyocr_model_identity()

    assert identity["easyocr_version"] == "1.7.2"
    assert identity["repository_relative_model_directory"] == (
        "cognitive/weights/easyocr/model"
    )
    assert identity["reader"] == {
        "languages": ["en"],
        "gpu": False,
        "detect_network": "craft",
        "recog_network": "standard",
        "download_enabled": False,
        "verbose": False,
        "quantize": True,
        "cudnn_benchmark": False,
    }
    assert identity["artifacts"]["detector"] == {
        "filename": "craft_mlt_25k.pth",
        "bytes": 83152330,
        "sha256": DETECTOR_SHA256,
        "upstream_md5": "2f8227d2def4037cdb3b34389dcf9ec1",
        "acquisition_url": (
            "https://github.com/JaidedAI/EasyOCR/releases/download/"
            "pre-v1.1.6/craft_mlt_25k.zip"
        ),
    }
    assert identity["artifacts"]["recognizer"] == {
        "filename": "english_g2.pth",
        "bytes": 15143997,
        "sha256": RECOGNIZER_SHA256,
        "upstream_md5": "5864788e1821be9e454ec108d61b887d",
        "acquisition_url": (
            "https://github.com/JaidedAI/EasyOCR/releases/download/"
            "v1.3/english_g2.zip"
        ),
    }


def test_exact_model_bytes_are_accepted(tmp_path, monkeypatch):
    identity_path, model_directory, expected = _synthetic_contract(tmp_path)
    monkeypatch.setattr(
        identity_module.importlib.metadata,
        "version",
        lambda name: "1.7.2" if name == "easyocr" else pytest.fail(name),
    )

    result = verify_easyocr_models(model_directory, identity_path=identity_path)

    assert result["model_directory"] == model_directory.resolve()
    for role in ("detector", "recognizer"):
        assert result["artifacts"][role]["sha256"] == expected["artifacts"][role][
            "sha256"
        ]


@pytest.mark.parametrize("role", ["detector", "recognizer"])
@pytest.mark.parametrize("fault", ["missing", "wrong-size", "same-size-tamper"])
def test_missing_substituted_or_tampered_model_fails_closed(
    tmp_path, monkeypatch, role, fault
):
    identity_path, model_directory, identity = _synthetic_contract(tmp_path)
    monkeypatch.setattr(
        identity_module.importlib.metadata, "version", lambda _name: "1.7.2"
    )
    artifact = identity["artifacts"][role]
    path = model_directory / artifact["filename"]
    if fault == "missing":
        path.unlink()
        match = "unavailable"
    elif fault == "wrong-size":
        path.write_bytes(path.read_bytes() + b"x")
        match = "byte-size mismatch"
    else:
        payload = bytearray(path.read_bytes())
        payload[0] ^= 0xFF
        path.write_bytes(payload)
        match = "SHA-256 mismatch"

    with pytest.raises(EasyOCRModelIntegrityError, match=match):
        verify_easyocr_models(model_directory, identity_path=identity_path)


def test_easyocr_package_version_mismatch_fails_before_artifact_use(
    tmp_path, monkeypatch
):
    identity_path, model_directory, _ = _synthetic_contract(tmp_path)
    monkeypatch.setattr(
        identity_module.importlib.metadata, "version", lambda _name: "9.9.9"
    )

    with pytest.raises(EasyOCRModelIntegrityError, match="package version"):
        verify_easyocr_models(model_directory, identity_path=identity_path)


def test_reader_verifies_before_import_and_uses_only_strict_fixed_arguments(
    monkeypatch, tmp_path
):
    identity = load_easyocr_model_identity()
    model_directory = tmp_path / "model"
    user_network_directory = tmp_path / "user_network"
    events = []

    def _verified():
        events.append("verify")
        return {
            "identity": identity,
            "model_directory": model_directory,
            "user_network_directory": user_network_directory,
            "artifacts": {},
        }

    expected_reader = object()
    calls = []

    def _reader(*args, **kwargs):
        events.append("reader")
        calls.append((args, kwargs))
        return expected_reader

    real_import = builtins.__import__

    def _tracking_import(name, *args, **kwargs):
        if name == "easyocr":
            events.append("import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(text_reader, "verify_easyocr_models", _verified)
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=_reader))
    monkeypatch.setattr(builtins, "__import__", _tracking_import)

    assert text_reader._get_reader() is expected_reader

    assert events == ["verify", "import", "reader"]
    assert calls == [
        (
            (["en"],),
            {
                "gpu": False,
                "model_storage_directory": str(model_directory),
                "user_network_directory": str(user_network_directory),
                "detect_network": "craft",
                "recog_network": "standard",
                "download_enabled": False,
                "verbose": False,
                "quantize": True,
                "cudnn_benchmark": False,
            },
        )
    ]


def test_failed_integrity_gate_prevents_easyocr_import(monkeypatch):
    imports = []
    real_import = builtins.__import__

    def _tracking_import(name, *args, **kwargs):
        if name == "easyocr":
            imports.append(name)
        return real_import(name, *args, **kwargs)

    def _reject():
        raise EasyOCRModelIntegrityError("tampered")

    monkeypatch.setattr(text_reader, "verify_easyocr_models", _reject)
    monkeypatch.setattr(builtins, "__import__", _tracking_import)

    assert text_reader._get_reader() is None
    assert imports == []
    assert text_reader._READER_FAILED is True
