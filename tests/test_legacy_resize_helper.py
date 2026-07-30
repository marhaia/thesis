"""Focused unit tests for the legacy-resize production helper.

Covers the ``LegacyBilinearUpSampling2D`` compatibility layer added to
``saliency.umsi_model`` as the narrow production fix:
  * non-square inputs,
  * odd source and target dimensions,
  * corner/boundary pixel behaviour,
  * dtype behaviour,
  * determinism (repeat runs are bit-identical),
  * equality with the verified legacy operator oracle
    (tf.raw_ops.ResizeBilinear, align_corners=False, half_pixel_centers=False),
  * the three actual decoder shapes (32->64, 64->128, 128->512),
  * config round-trip (serialisation), and a structural check that exactly the
    three decoder call sites use the helper.

These tests require TensorFlow and run in the modern (TF2.16/Keras3) runtime.
"""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from keras import layers  # noqa: E402

from saliency.umsi_model import (  # noqa: E402
    LegacyBilinearUpSampling2D, build_umsi_model)


def _legacy_rawop(x, fh, fw):
    sh = tf.shape(x)
    tgt = tf.stack([sh[1] * fh, sh[2] * fw])
    return tf.raw_ops.ResizeBilinear(
        images=x, size=tgt, align_corners=False,
        half_pixel_centers=False).numpy()


def _apply(arr, fh, fw):
    return LegacyBilinearUpSampling2D(size=(fh, fw))(tf.constant(arr)).numpy()


@pytest.mark.parametrize("h,w,c,fh,fw", [
    (5, 7, 3, 2, 2),     # non-square, prime dims
    (3, 9, 1, 4, 2),     # non-square factors
    (7, 7, 2, 3, 3),     # odd square, odd factor -> odd target
    (2, 3, 4, 2, 4),     # tiny, many channels
])
def test_matches_legacy_operator_generic(h, w, c, fh, fw):
    rng = np.random.default_rng(0)
    arr = rng.random((1, h, w, c), dtype=np.float32)
    out = _apply(arr, fh, fw)
    ref = _legacy_rawop(tf.constant(arr), fh, fw)
    assert out.shape == (1, h * fh, w * fw, c)
    assert np.array_equal(out, ref), "helper must equal legacy operator bit-for-bit"


@pytest.mark.parametrize("name,h,w,c,fh,fw,oh,ow", [
    ("dec_ups1", 32, 32, 256, 2, 2, 64, 64),
    ("dec_ups2", 64, 64, 128, 2, 2, 128, 128),
    ("dec_ups3", 128, 128, 64, 4, 4, 512, 512),
])
def test_actual_decoder_shapes(name, h, w, c, fh, fw, oh, ow):
    rng = np.random.default_rng(1)
    arr = rng.random((1, h, w, c), dtype=np.float32)
    out = _apply(arr, fh, fw)
    ref = _legacy_rawop(tf.constant(arr), fh, fw)
    assert out.shape == (1, oh, ow, c)
    assert np.array_equal(out, ref)
    assert np.isfinite(out).all()


def test_boundary_corner_pixels_preserved():
    # align_corners=False: top-left source pixel maps to top-left output pixel.
    arr = np.zeros((1, 4, 6, 1), dtype=np.float32)
    arr[0, 0, 0, 0] = 1.0
    arr[0, -1, -1, 0] = 2.0
    out = _apply(arr, 2, 2)
    assert out[0, 0, 0, 0] == pytest.approx(1.0)
    assert out[0, -1, -1, 0] == pytest.approx(2.0)


def test_odd_target_dimensions():
    arr = np.arange(1 * 3 * 5 * 1, dtype=np.float32).reshape(1, 3, 5, 1)
    out = _apply(arr, 3, 3)  # -> 9 x 15 (odd)
    assert out.shape == (1, 9, 15, 1)
    assert np.array_equal(out, _legacy_rawop(tf.constant(arr), 3, 3))


def test_dtype_float32_output():
    arr = np.random.default_rng(2).random((1, 4, 4, 3)).astype(np.float32)
    out = _apply(arr, 2, 2)
    assert out.dtype == np.float32


def test_determinism_repeat():
    arr = np.random.default_rng(3).random((1, 8, 5, 2)).astype(np.float32)
    a = _apply(arr, 4, 2)
    b = _apply(arr, 4, 2)
    assert np.array_equal(a, b)


def test_int_size_argument_equivalent_to_tuple():
    arr = np.random.default_rng(4).random((1, 6, 6, 1)).astype(np.float32)
    out_int = LegacyBilinearUpSampling2D(size=2)(tf.constant(arr)).numpy()
    out_tuple = _apply(arr, 2, 2)
    assert np.array_equal(out_int, out_tuple)


def test_get_config_round_trip():
    layer = LegacyBilinearUpSampling2D(size=(4, 4), name="rt")
    cfg = layer.get_config()
    assert tuple(cfg["size"]) == (4, 4)
    clone = LegacyBilinearUpSampling2D.from_config(cfg)
    arr = np.random.default_rng(5).random((1, 5, 7, 3)).astype(np.float32)
    assert np.array_equal(_apply(arr, 4, 4),
                          clone(tf.constant(arr)).numpy())


def test_compute_output_shape():
    layer = LegacyBilinearUpSampling2D(size=(2, 4))
    assert tuple(layer.compute_output_shape((None, 16, 8, 32))) == \
        (None, 32, 32, 32)


def test_differs_from_halfpixel_upsampling():
    # The fix must actually change behaviour relative to the op it replaces.
    arr = np.random.default_rng(6).random((1, 8, 8, 3)).astype(np.float32)
    legacy = _apply(arr, 2, 2)
    halfpix = layers.UpSampling2D(size=(2, 2),
                                  interpolation="bilinear")(
        tf.constant(arr)).numpy()
    assert not np.array_equal(legacy, halfpix)


def test_structural_exactly_three_call_sites():
    model = build_umsi_model(verbose=False)
    legacy = {l.name for l in model.layers
              if isinstance(l, LegacyBilinearUpSampling2D)}
    stock = [l.name for l in model.layers
             if type(l).__name__ == "UpSampling2D"]
    assert legacy == {"dec_ups1", "dec_ups2", "dec_ups3"}
    assert stock == []
