"""Regression tests for the production UMSI postprocessor."""

import numpy as np
import pytest

from saliency.postprocessing import normalize_saliency_map, postprocess_saliency
from saliency.saliency_features import extract_saliency_features


@pytest.mark.parametrize(
    "pred",
    [
        np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32),
        np.array([[10.0, 11.0], [14.0, 18.0]], dtype=np.float32),
        np.array([[-8.0, -4.0], [-2.0, -1.0]], dtype=np.float32),
    ],
    ids=["normal", "positive-offset", "negative"],
)
def test_production_postprocessor_uses_true_minmax(pred):
    observed = postprocess_saliency(pred, *pred.shape)
    expected = (pred.astype(np.float64) - pred.min()) / (pred.max() - pred.min())

    assert observed.dtype == np.float32
    assert observed.shape == pred.shape
    assert np.isfinite(observed).all()
    assert observed.min() == pytest.approx(0.0)
    assert observed.max() == pytest.approx(1.0)
    np.testing.assert_allclose(observed, expected.astype(np.float32), rtol=0, atol=0)


@pytest.mark.parametrize("value", [0.0, 7.5, -3.0])
def test_constant_map_policy_is_deterministic_zero_map(value):
    pred = np.full((3, 4), value, dtype=np.float32)

    observed = postprocess_saliency(pred, 3, 4)

    assert observed.dtype == np.float32
    assert observed.shape == (3, 4)
    assert np.array_equal(observed, np.zeros((3, 4), dtype=np.float32))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_non_finite_predictions_fail_closed(invalid):
    pred = np.full((2, 2), invalid, dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        postprocess_saliency(pred, 2, 2)


def test_singleton_channel_and_resize_preserve_output_contract():
    pred = np.arange(12, dtype=np.float32).reshape(3, 4, 1)

    observed = postprocess_saliency(pred, 6, 5)

    assert observed.dtype == np.float32
    assert observed.shape == (6, 5)
    assert np.isfinite(observed).all()
    assert observed.min() == pytest.approx(0.0)
    assert observed.max() == pytest.approx(1.0)


def test_multichannel_prediction_fails_closed():
    with pytest.raises(ValueError, match="one channel"):
        postprocess_saliency(np.zeros((2, 2, 2), dtype=np.float32), 2, 2)


@pytest.mark.parametrize(
    "raw_map",
    [
        np.array([[10.0, 11.0], [14.0, 18.0]], dtype=np.float32),
        np.array([[-8.0, -4.0], [-2.0, -1.0]], dtype=np.float32),
    ],
    ids=["positive-offset", "negative"],
)
def test_feature_extractor_uses_the_production_normalization_policy(raw_map):
    expected = extract_saliency_features(normalize_saliency_map(raw_map))

    observed = extract_saliency_features(raw_map)

    assert observed.keys() == expected.keys()
    for name in observed:
        assert observed[name] == pytest.approx(expected[name], rel=0, abs=0)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_feature_extractor_rejects_non_finite_maps(invalid):
    saliency_map = np.array([[0.0, 1.0], [2.0, invalid]], dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        extract_saliency_features(saliency_map)


def test_finite_extreme_range_is_normalized_without_overflow():
    limit = np.finfo(np.float64).max
    saliency_map = np.array([[-limit, 0.0, limit]], dtype=np.float64)

    observed = normalize_saliency_map(saliency_map)

    np.testing.assert_array_equal(
        observed,
        np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
    )
