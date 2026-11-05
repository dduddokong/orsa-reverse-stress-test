"""1단계 부채 현금흐름 모형 검사. 모델포인트는 검사용 합성값이다."""
from dataclasses import replace

import pytest

from model.liability import (
    Assumptions,
    ModelPoint,
    bel,
    long_risk,
    own_funds_contribution,
    portfolio_bel,
    risk_margin,
    sub_risks,
)

A0 = Assumptions()
MPS = [
    ModelPoint("health_paying", term=40, pay_years=20, premium=100.0, risk_share=0.3),
    ModelPoint("health_paid", term=30, pay_years=0, premium=80.0, risk_share=0.6),
]


def test_profitable_block_has_negative_bel():
    assert bel(MPS[0], A0) < 0


def test_higher_loss_ratio_raises_bel():
    assert portfolio_bel(MPS, replace(A0, loss_ratio=0.95)) > portfolio_bel(MPS, A0)


def test_paid_up_block_is_a_pure_liability():
    assert bel(MPS[1], A0) > 0


def test_sub_risks_are_nonnegative_and_disease_dominates_for_health():
    s = sub_risks(MPS, A0)
    assert all(v >= 0 for v in s.values())
    assert s["disease"] > s["death"]


def test_diversified_long_risk_is_below_sum():
    s = sub_risks(MPS, A0)
    assert long_risk(s) < sum(s.values())


def test_risk_margin_reduces_own_funds():
    s = sub_risks(MPS, A0)
    assert own_funds_contribution(MPS, A0) == pytest.approx(-portfolio_bel(MPS, A0) - risk_margin(s))


def test_lapse_rise_hurts_profitable_block_more_than_fall():
    up = portfolio_bel(MPS[:1], replace(A0, lapse_pay=A0.lapse_pay * 1.35))
    down = portfolio_bel(MPS[:1], replace(A0, lapse_pay=A0.lapse_pay * 0.65))
    assert up > down


def test_vectorised_block_matches_reference():
    import numpy as np

    from model.liability import CORR_LONG4, RM_FACTOR, SUB_RISKS, LiabilityBlock

    a2 = replace(A0, lapse_pay=0.1, expense_ratio=0.12)
    block = LiabilityBlock(((tuple(MPS), A0), (tuple(MPS[:1]), a2)))
    subs = np.zeros(4)
    total_bel = 0.0
    for mps, a in block.blocks:
        s = sub_risks(mps, a)
        subs += np.array([s[k] for k in SUB_RISKS])
        total_bel += portfolio_bel(mps, a)
    long = float(np.sqrt(subs @ CORR_LONG4 @ subs))
    of, lr = block.evaluate()
    assert lr == pytest.approx(long, rel=1e-10)
    assert of == pytest.approx(-(total_bel + long * RM_FACTOR), rel=1e-10)
