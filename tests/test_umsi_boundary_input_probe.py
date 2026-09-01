import numpy as np
import pytest

from stage1.tools.umsi_boundary_input_probe import (
    build_candidates_from_padded,
    capture_padding_and_predict_once,
    capture_predict_once,
)


class DummyOwner:
    def __init__(self):
        self.calls = 0

        def _predict(x, *args, **kwargs):
            self.calls += 1
            return {"x": np.array(x, copy=True)}

        self.predict = _predict


class RaisingOwner:
    def __init__(self):
        self.calls = 0

        def _predict(x, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("boom")

        self.predict = _predict


class PaddingModule:
    def __init__(self):
        self.calls = 0

        def _padding(x, *args, **kwargs):
            self.calls += 1
            return np.array(x, copy=True)

        self._padding = _padding


class RaisingPaddingModule:
    def __init__(self):
        self.calls = 0

        def _padding(x, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("padding boom")

        self._padding = _padding


def test_proxy_restored_after_success():
    owner = DummyOwner()
    original = owner.predict

    capture, invoke_result = capture_predict_once(
        owner,
        lambda: owner.predict(np.ones((1, 2, 3), dtype=np.float32), verbose=0),
    )

    assert capture.call_count == 1
    assert capture.restored is True
    assert owner.predict is original
    assert owner.calls == 1
    assert capture.input_array.shape == (1, 2, 3)
    assert capture.input_array.dtype == np.float32
    assert invoke_result["x"].shape == (1, 2, 3)


def test_proxy_restored_after_exception():
    owner = RaisingOwner()
    original = owner.predict

    with pytest.raises(RuntimeError, match="boom"):
        capture_predict_once(
            owner,
            lambda: owner.predict(np.ones((1, 2, 3), dtype=np.float32), verbose=0),
        )

    assert owner.predict is original
    assert owner.calls == 1


def test_build_candidates_from_padded_expected_values():
    padded = np.full((256, 256, 3), 200, dtype=np.uint8)
    current_float32, legacy_then_float32 = build_candidates_from_padded(padded)

    assert current_float32.shape == (1, 256, 256, 3)
    assert legacy_then_float32.shape == (1, 256, 256, 3)
    assert current_float32.dtype == np.float32
    assert legacy_then_float32.dtype == np.float32
    assert np.isfinite(current_float32).all()
    assert np.isfinite(legacy_then_float32).all()
    assert current_float32[0, 0, 0, 0] == pytest.approx(200.0 - 103.939)
    assert current_float32[0, 0, 0, 1] == pytest.approx(200.0 - 116.779)
    assert current_float32[0, 0, 0, 2] == pytest.approx(200.0 - 123.68)


def test_build_candidates_from_padded_rejects_bad_shape_dtype():
    with pytest.raises(ValueError):
        build_candidates_from_padded(np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        build_candidates_from_padded(np.zeros((256, 256, 3), dtype=np.float32))


def test_dual_capture_restored_after_success():
    mod = PaddingModule()
    owner = DummyOwner()
    orig_padding = mod._padding
    orig_predict = owner.predict

    sample = np.ones((1, 2, 3), dtype=np.float32)

    def invoke():
        _ = mod._padding(np.ones((256, 256, 3), dtype=np.uint8), 256, 256)
        return owner.predict(sample, verbose=0)

    cap, out = capture_padding_and_predict_once(mod, owner, invoke)
    assert cap.padding_call_count == 1
    assert cap.predict_call_count == 1
    assert cap.padding_restored is True
    assert cap.predict_restored is True
    assert mod._padding is orig_padding
    assert owner.predict is orig_predict
    assert cap.padded_uint8.shape == (256, 256, 3)
    assert cap.padded_uint8.dtype == np.uint8
    assert cap.predict_input.shape == (1, 2, 3)
    assert cap.predict_input.dtype == np.float32
    assert out["x"].shape == (1, 2, 3)


def test_dual_capture_restored_after_exception():
    mod = RaisingPaddingModule()
    owner = DummyOwner()
    orig_padding = mod._padding
    orig_predict = owner.predict

    def invoke():
        _ = mod._padding(np.ones((256, 256, 3), dtype=np.uint8), 256, 256)
        return owner.predict(np.ones((1, 2, 3), dtype=np.float32), verbose=0)

    with pytest.raises(RuntimeError, match="padding boom"):
        capture_padding_and_predict_once(mod, owner, invoke)

    assert mod._padding is orig_padding
    assert owner.predict is orig_predict


def test_preprocess_image_matches_legacy_float64_order(monkeypatch):
    import importlib

    umsi_mod = importlib.import_module("saliency.umsi_model")

    # Deterministic varied uint8 values spanning full dynamic range.
    padded = np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3)
    padded = (padded % 256).astype(np.uint8)
    fake_image = np.full((20, 30, 3), 17, dtype=np.uint8)

    monkeypatch.setattr(umsi_mod.cv2, "imread", lambda _: fake_image)
    monkeypatch.setattr(umsi_mod, "_padding", lambda *_args, **_kwargs: padded)

    observed = umsi_mod.preprocess_image("ignored.png", shape_r=256, shape_c=256)

    expected_legacy = np.zeros((1, 256, 256, 3), dtype=np.float64)
    expected_legacy[0] = padded
    expected_legacy[..., 0] -= umsi_mod.VGG_MEAN_B
    expected_legacy[..., 1] -= umsi_mod.VGG_MEAN_G
    expected_legacy[..., 2] -= umsi_mod.VGG_MEAN_R
    expected_legacy = expected_legacy.astype(np.float32)

    rejected_current = padded.astype(np.float32)
    rejected_current[..., 0] -= umsi_mod.VGG_MEAN_B
    rejected_current[..., 1] -= umsi_mod.VGG_MEAN_G
    rejected_current[..., 2] -= umsi_mod.VGG_MEAN_R
    rejected_current = np.expand_dims(rejected_current, axis=0)

    assert list(observed.shape) == [1, 256, 256, 3]
    assert observed.dtype == np.float32
    assert np.isfinite(observed).all()
    assert np.array_equal(observed, expected_legacy)
    assert not np.array_equal(observed, rejected_current)
