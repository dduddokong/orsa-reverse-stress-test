"""results/REGISTRY.md에 등록한 A-### 값이 현재 모형과 맞는지 본다.

모형을 바꿔 이 테스트가 깨지면, 수치를 다시 산출해 REGISTRY와 원고를 함께 고친다(값을 테스트에서만 고치지 않는다).
"""
import numpy as np
import pytest

from interfaces.contracts import N_FACTORS, NO_ACTION, Regime
from model.archetypes import build_all
from model.backtest import factor_change, macro_contribution
from model.insurer import StylizedInsurer

ZERO = np.zeros(N_FACTORS)


@pytest.fixture(scope="module")
def models():
    return {k: StylizedInsurer(p) for k, p in build_all().items()}


def _calm(m):
    return m.step(m.initial(), ZERO, Regime.NONE, NO_ACTION)


def test_a002_a008_starting_ratios(models):
    expected = {"A": (2.828, 2.199), "B": (2.044, 0.960), "C": (2.314, 1.316), "D": (1.500, 0.620)}
    for k, (ratio, basic) in expected.items():
        b0 = models[k].initial()
        assert b0.ratio == pytest.approx(ratio, abs=5e-4)
        assert b0.basic_ratio == pytest.approx(basic, abs=5e-4)


def test_a004_rate_response(models):
    m = models["B"]
    x = ZERO.copy()
    x[0] = 0.01
    assert m.step(m.initial(), x, Regime.NONE, NO_ACTION).available_capital - _calm(m).available_capital == pytest.approx(
        -0.893, abs=5e-3
    )


def test_a005_mass_lapse(models):
    m = models["B"]
    x = ZERO.copy()
    x[5] = 0.25 / m.p.liability.first_year_lapse
    assert m.step(m.initial(), x, Regime.NONE, NO_ACTION).available_capital - _calm(m).available_capital == pytest.approx(
        -4.52, abs=0.01
    )


def test_a006_backtest_rates(models):
    start = dict(ktb3=2.953, ktb10=3.385, aa3=3.476, kospi=4214.17)
    end = dict(ktb3=3.703, ktb10=4.091, aa3=4.375, kospi=8476.48)
    x = factor_change(start, end)
    x[3] = 0.0
    assert macro_contribution(models["A"].p, x) == pytest.approx(-6.6, abs=0.05)


def test_a009_lapse_reform(models):
    m = models["B"]
    j = m.step(m.initial(), ZERO, Regime.LAPSE, NO_ACTION)
    assert 100 * (j.ratio - _calm(m).ratio) == pytest.approx(-21.5, abs=0.05)


def test_a010_loss_reform_ratio(models):
    m = models["A"]
    j = m.step(m.initial(), ZERO, Regime.LOSS, NO_ACTION)
    assert 100 * (j.ratio - _calm(m).ratio) == pytest.approx(-3.8, abs=0.05)


def test_a007_regime_sizes():
    from model.archetypes import build_all

    expected = {"A": (2.3, 0.533), "B": (1.52, 0.524), "C": (1.68, 0.534), "D": (1.10, 0.556)}
    for k, p in build_all().items():
        d_lr, lapse = expected[k]
        assert 100 * p.regime_shift[Regime.LOSS][0] == pytest.approx(d_lr, abs=0.01)
        assert p.regime_shift[Regime.LAPSE][2] == pytest.approx(lapse, abs=0.002)
