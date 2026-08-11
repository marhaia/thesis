"""P6 checkpoint-identity regression tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage1"))

from saliency.checkpoint_identity import (  # noqa: E402
    CheckpointIntegrityError,
    IDENTITY_PATH,
    load_umsi_checkpoint_identity,
    verify_checkpoint,
)


def _small_identity(content: bytes, artifact: str = "umsi++.hdf5") -> dict:
    return {
        "artifact": artifact,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_frozen_checkpoint_identity_matches_p2_evidence_contract():
    identity = load_umsi_checkpoint_identity()
    contract_path = ROOT / "stage1" / "evidence" / "umsi_five_fixture_gate_contract.json"
    with contract_path.open(encoding="utf-8") as source:
        contract = json.load(source)

    assert identity["artifact"] == contract["weights_filename"] == "umsi++.hdf5"
    assert identity["bytes"] == contract["weights_size_bytes"] == 120093896
    assert identity["sha256"] == contract["weights_sha256"] == (
        "f4290c3f11f18befbb47de50d81e4555ec8e7a63066c71c343a32fe32799e9fe"
    )
    assert identity["claim_boundary"]


def test_checkpoint_verifier_accepts_exact_bytes(tmp_path):
    content = b"p6-exact-checkpoint-fixture"
    checkpoint = tmp_path / "umsi++.hdf5"
    checkpoint.write_bytes(content)

    result = verify_checkpoint(checkpoint, identity=_small_identity(content))

    assert result == {
        "artifact": "umsi++.hdf5",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


@pytest.mark.parametrize("failure", ["missing", "filename", "size", "sha256"])
def test_checkpoint_verifier_fails_closed_on_identity_mismatch(tmp_path, failure):
    content = b"p6-checkpoint"
    checkpoint = tmp_path / "umsi++.hdf5"
    identity = _small_identity(content)

    if failure == "missing":
        expected = "unavailable"
    else:
        checkpoint.write_bytes(content)
        expected = "mismatch"
        if failure == "filename":
            checkpoint = checkpoint.with_name("other.hdf5")
            checkpoint.write_bytes(content)
        elif failure == "size":
            identity["bytes"] += 1
        elif failure == "sha256":
            identity["sha256"] = "0" * 64

    with pytest.raises(CheckpointIntegrityError, match=expected):
        verify_checkpoint(checkpoint, identity=identity)


def test_production_verifies_checkpoint_before_importing_tensorflow(monkeypatch):
    import app as app_module

    app_module._saliency_model = None
    verified = []

    def _reject_before_model_import(path):
        verified.append(Path(path).name)
        raise CheckpointIntegrityError("test identity rejection")

    monkeypatch.setattr(app_module, "verify_umsi_checkpoint", _reject_before_model_import)
    sys.modules.pop("saliency.umsi_model", None)

    with pytest.raises(CheckpointIntegrityError, match="test identity rejection"):
        app_module._get_saliency_model()

    assert verified == ["umsi++.hdf5"]
    assert "saliency.umsi_model" not in sys.modules


def test_identity_manifest_schema_fails_closed(tmp_path):
    malformed = tmp_path / "identity.json"
    malformed.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="invalid schema"):
        load_umsi_checkpoint_identity(malformed)


def test_downloader_requires_exact_verification_before_reusing_checkpoint(
    tmp_path, monkeypatch
):
    from scripts import download_weights

    checkpoint = tmp_path / "umsi++.hdf5"
    checkpoint.write_bytes(b"existing-checkpoint")
    verified = []

    def _verify(path):
        verified.append(Path(path))
        return {
            "artifact": "umsi++.hdf5",
            "bytes": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }

    def _unexpected_download():
        raise AssertionError("a verified existing checkpoint must not be downloaded again")

    monkeypatch.setattr(download_weights, "TARGET_PATH", checkpoint)
    monkeypatch.setattr(download_weights, "verify_umsi_checkpoint", _verify)
    monkeypatch.setattr(download_weights, "_build_request", _unexpected_download)

    assert download_weights.main() == 0
    assert verified == [checkpoint]
