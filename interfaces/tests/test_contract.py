"""KicsModel 구현이 모두 지켜야 하는 계약 테스트."""
import numpy as np
import pytest

from interfaces.contracts import (
    FACTORS,
    N_FACTORS,
    NO_ACTION,
    BalanceSheet,
    KicsModel,
    Regime,
)

ZERO = np.zeros(N_FACTORS)


def shock(name, size):
    x = np.zeros(N_FACTORS)
    x[FACTORS.index(name)] = size
    return x


def test_implements_protocol(model):
    assert isinstance(model, KicsModel)


def test_initial_ratio_plausible(model):
    b0 = model.initial()
    assert isinstance(b0, BalanceSheet)
    assert b0.t == 0
    assert 1.0 < b0.ratio < 4.0


def test_step_advances_time_and_stays_finite(model):
    b1 = model.step(model.initial(), ZERO, Regime.NONE, NO_ACTION)
    assert isinstance(b1, BalanceSheet)
    assert b1.t == 1
    assert b1.required_capital > 0
    assert np.isfinite(b1.available_capital)


def test_step_is_deterministic(model):
    b0 = model.initial()
    x = shock("loss_ratio", 0.05)
    assert model.step(b0, x, Regime.LOSS, NO_ACTION) == model.step(b0, x, Regime.LOSS, NO_ACTION)


def test_baseline_year_is_calm(model):
    b0 = model.initial()
    b1 = model.step(b0, ZERO, Regime.NONE, NO_ACTION)
    assert abs(b1.ratio - b0.ratio) < 0.3


@pytest.mark.parametrize(
    "name,size",
    [("loss_ratio", 0.05), ("credit_spread", 0.01), ("expense_infl", 0.02)],
)
def test_adverse_shock_lowers_ratio(model, name, size):
    b0 = model.initial()
    base = model.step(b0, ZERO, Regime.NONE, NO_ACTION).ratio
    hit = model.step(b0, shock(name, size), Regime.NONE, NO_ACTION).ratio
    assert hit < base


@pytest.mark.parametrize("regime", [Regime.LOSS, Regime.LAPSE, Regime.BOTH])
def test_regime_jump_lowers_ratio(model, regime):
    b0 = model.initial()
    base = model.step(b0, ZERO, Regime.NONE, NO_ACTION).ratio
    assert model.step(b0, ZERO, regime, NO_ACTION).ratio < base


def test_feasible_actions_include_no_action(model):
    b0 = model.initial()
    for x in (ZERO, shock("credit_spread", 0.03)):
        assert NO_ACTION in list(model.feasible_actions(b0, x))


def test_rejects_wrong_shape(model):
    with pytest.raises(ValueError):
        model.step(model.initial(), np.zeros(N_FACTORS + 1), Regime.NONE, NO_ACTION)
