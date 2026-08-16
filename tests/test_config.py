import pytest

from core.config import DEFAULT_WEIGHTS, FACTOR_LABELS, STRATEGY_DESCRIPTIONS, STRATEGY_PRESETS, NeedWeights


def test_default_weights_have_labels():
    # If someone adds a factor to one table without the other, the UI
    # would silently show a raw key instead of a readable label.
    assert set(DEFAULT_WEIGHTS.keys()) == set(FACTOR_LABELS.keys())


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_high_weight_outweighs_low_after_normalization():
    weights = NeedWeights(weights={"no_vehicle": 0.30, "uninsured": 0.05})
    normalized = weights.normalized()
    assert normalized["no_vehicle"] > normalized["uninsured"]
    assert normalized["no_vehicle"] == pytest.approx(6 * normalized["uninsured"])


def test_all_zero_normalizes_without_dividing_by_zero():
    weights = NeedWeights(weights={"no_vehicle": 0.0, "poverty": 0.0})
    assert weights.normalized() == {"no_vehicle": 0.0, "poverty": 0.0}


def test_every_strategy_preset_has_a_description():
    assert set(STRATEGY_PRESETS.keys()) == set(STRATEGY_DESCRIPTIONS.keys())


def test_weighted_strategy_presets_cover_every_factor_and_sum_to_one():
    for name, weights in STRATEGY_PRESETS.items():
        if weights is None:
            continue  # "Maximize total residents covered" -- pure population mode
        assert set(weights.keys()) == set(DEFAULT_WEIGHTS.keys()), name
        assert sum(weights.values()) == pytest.approx(1.0), name
