"""모형 엔진 테스트. 모수는 검사용 합성값이며 결과에 쓰지 않는다."""
import numpy as np
import pytest

from interfaces.contracts import FACTORS, N_FACTORS, NO_ACTION, Action, BalanceSheet, Regime
from model.insurer import (
    SUPPLEMENTARY_LIMIT,
    InsurerParams,
    InsurerState,
    StylizedInsurer,
    Transitional,
    describe,
)
from model.kics import aggregate, check_corr, required_capital

ZERO = np.zeros(N_FACTORS)
CORR = np.array(
    [
        [1.0, 0.0, 0.25, 0.25],
        [0.0, 1.0, 0.25, 0.25],
        [0.25, 0.25, 1.0, 0.25],
        [0.25, 0.25, 0.25, 1.0],
    ]
)


def factor(name, size):
    x = np.zeros(N_FACTORS)
    x[FACTORS.index(name)] = size
    return x


def synthetic(**overrides):
    risk_sens = np.zeros((4, N_FACTORS))
    risk_sens[0, FACTORS.index("loss_ratio")] = 2.0
    risk_sens[2, FACTORS.index("equity")] = 0.5
    kw = dict(
        name="synthetic",
        base_year=2026,
        basic_capital=8.0,
        supplementary_capital=0.5,  # 한도(0.5 × RC) 아래
        risks=np.array([3.0, 1.0, 2.0, 0.5]),
        corr=CORR,
        operational=0.3,
        tax_rate=0.2,
        tax_cap=10.0,
        ac_growth=0.06,
        rc_growth=0.03,
        dividend_payout=0.02,
        ac_sens=np.array([3.0, 0.5, -5.0, 0.2, -2.0, -0.1, -2.0]),
        ac_rate_convexity=-20.0,
        risk_sens=risk_sens,
        regime_ac_hit={Regime.NONE: 0.0, Regime.LOSS: 0.06, Regime.LAPSE: 0.04, Regime.BOTH: 0.10},
        regime_risk_rise={
            Regime.NONE: np.zeros(4),
            Regime.LOSS: np.array([0.05, 0, 0, 0]),
            Regime.LAPSE: np.array([0.03, 0, 0, 0]),
            Regime.BOTH: np.array([0.08, 0, 0, 0]),
        },
    )
    kw.update(overrides)
    return InsurerParams(**kw)


def test_aggregate_matches_independent_and_comonotone_limits():
    r = np.array([3.0, 4.0, 0.0, 0.0])
    assert aggregate(r, np.eye(4)) == pytest.approx(5.0)
    assert aggregate(r, np.ones((4, 4))) == pytest.approx(7.0)


def test_check_corr_rejects_bad_matrix():
    with pytest.raises(ValueError):
        check_corr(np.full((4, 4), 2.0))


def test_tax_adjustment_scales_with_risk_until_cap():
    r = np.array([3.0, 1.0, 2.0, 0.5])
    small = required_capital(r, CORR, 0.3, 0.2, 100.0)
    capped = required_capital(r, CORR, 0.3, 0.2, 0.1)
    basic = aggregate(r, CORR) + 0.3
    assert small == pytest.approx(basic * 0.8)
    assert capped == pytest.approx(basic - 0.1)


def test_describe_reports_positive_diversification():
    d = describe(synthetic())
    assert d["diversification"] > 0
    assert d["ratio"] == pytest.approx(d["available_capital"] / d["required_capital"])


def test_regime_table_must_be_complete():
    with pytest.raises(ValueError):
        synthetic(regime_ac_hit={Regime.NONE: 0.0})


def test_supplementary_capital_capped_at_half_of_required():
    m = StylizedInsurer(synthetic(supplementary_capital=50.0))
    b0 = m.initial()
    assert b0.available_capital == pytest.approx(8.0 + SUPPLEMENTARY_LIMIT * b0.required_capital)


def test_sub_debt_adds_nothing_beyond_the_cap():
    m = StylizedInsurer(synthetic(supplementary_capital=50.0))
    b0 = m.initial()
    plain = m.step(b0, ZERO, Regime.NONE, Action(dividend_cut=True))
    issued = m.step(b0, ZERO, Regime.NONE, Action(sub_debt=0.05, dividend_cut=True))
    assert issued.available_capital == pytest.approx(plain.available_capital)
    assert all(a.sub_debt == 0 for a in m.feasible_actions(b0, ZERO))


def test_transitional_phase_in_raises_required_capital_without_shocks():
    phased = Transitional(
        phased_share=np.array([0.5, 0.0, 0.5, 0.0]),
        insurance_schedule={2026: 0.3, 2027: 0.4},
        market_schedule={2026: 0.72, 2027: 0.76},
    )
    flat = StylizedInsurer(synthetic(rc_growth=0.0))
    ramp = StylizedInsurer(synthetic(rc_growth=0.0, transitional=phased))
    g_flat = flat.step(flat.initial(), ZERO, Regime.NONE, NO_ACTION).required_capital / flat.initial().required_capital
    g_ramp = ramp.step(ramp.initial(), ZERO, Regime.NONE, NO_ACTION).required_capital / ramp.initial().required_capital
    assert g_ramp > g_flat


def test_reinsurance_lowers_required_capital():
    m = StylizedInsurer(synthetic())
    b0 = m.initial()
    plain = m.step(b0, ZERO, Regime.NONE, NO_ACTION)
    ceded = m.step(b0, ZERO, Regime.NONE, Action(reinsurance_cession=0.2))
    assert ceded.required_capital < plain.required_capital


def test_reinsurance_costs_more_when_loss_ratio_rises():
    m = StylizedInsurer(synthetic())
    b0 = m.initial()
    x = factor("loss_ratio", 0.1)
    a = Action(reinsurance_cession=0.2)

    def cost(x):
        return (
            m.step(b0, x, Regime.NONE, NO_ACTION).available_capital
            - m.step(b0, x, Regime.NONE, a).available_capital
        )

    assert cost(x) > cost(ZERO)


def test_sub_debt_closed_when_spreads_blow_out():
    m = StylizedInsurer(synthetic())
    b0 = m.initial()
    assert any(a.sub_debt > 0 for a in m.feasible_actions(b0, ZERO))
    assert all(a.sub_debt == 0 for a in m.feasible_actions(b0, factor("credit_spread", 0.03)))


def test_regime_effect_persists_through_state():
    m = StylizedInsurer(synthetic())
    b0 = m.initial()
    hit = m.step(b0, ZERO, Regime.LOSS, NO_ACTION)
    calm = m.step(b0, ZERO, Regime.NONE, NO_ACTION)
    assert m.step(hit, ZERO, Regime.NONE, NO_ACTION).ratio < m.step(calm, ZERO, Regime.NONE, NO_ACTION).ratio


def test_risk_mix_changes_are_carried_forward():
    m = StylizedInsurer(synthetic())
    b1 = m.step(m.initial(), factor("loss_ratio", 0.1), Regime.NONE, NO_ACTION)
    assert isinstance(b1, InsurerState)
    share0 = m.initial().risks[0] / sum(m.initial().risks)
    assert b1.risks[0] / sum(b1.risks) > share0


def test_plain_balance_sheet_is_accepted():
    m = StylizedInsurer(synthetic())
    b0 = m.initial()
    plain = BalanceSheet(b0.available_capital, b0.required_capital, 0)
    assert m.step(plain, ZERO, Regime.NONE, NO_ACTION).ratio == pytest.approx(
        m.step(b0, ZERO, Regime.NONE, NO_ACTION).ratio
    )


def test_scheduled_change_hits_only_its_year():
    m = StylizedInsurer(synthetic(scheduled_ac_hit={2028: 0.05}))
    plain = StylizedInsurer(synthetic())
    b, q = m.initial(), plain.initial()
    gaps = []
    for _ in range(3):
        b, q = m.step(b, ZERO, Regime.NONE, NO_ACTION), plain.step(q, ZERO, Regime.NONE, NO_ACTION)
        gaps.append(q.available_capital - b.available_capital)
    assert gaps[0] == pytest.approx(0.0)
    assert gaps[1] > 0


def test_derisking_lowers_next_year_equity_sensitivity():
    m = StylizedInsurer(synthetic())
    b0 = m.initial()
    crash = factor("equity", -0.3)
    kept = m.step(b0, ZERO, Regime.NONE, NO_ACTION)
    sold = m.step(b0, ZERO, Regime.NONE, Action(derisk=0.5))
    loss_kept = kept.available_capital - m.step(kept, crash, Regime.NONE, NO_ACTION).available_capital
    loss_sold = sold.available_capital - m.step(sold, crash, Regime.NONE, NO_ACTION).available_capital
    assert sold.equity_scale == pytest.approx(0.5)
    assert loss_sold / sold.available_capital < loss_kept / kept.available_capital


def test_cession_persists_as_lower_retention():
    m = StylizedInsurer(synthetic())
    b1 = m.step(m.initial(), ZERO, Regime.NONE, Action(reinsurance_cession=0.2))
    b2 = m.step(b1, ZERO, Regime.NONE, Action(reinsurance_cession=0.2))
    assert b2.retention == pytest.approx(0.64)


def test_strategic_holdings_block_derisking():
    from model.insurer import ActionTerms

    m = StylizedInsurer(synthetic(actions=ActionTerms(min_equity_scale=0.9)))
    assert all(a.derisk == 0 for a in m.feasible_actions(m.initial(), ZERO))


def test_equity_exposure_follows_market_value():
    m = StylizedInsurer(synthetic())
    b1 = m.step(m.initial(), factor("equity", np.log(0.5)), Regime.NONE, NO_ACTION)
    assert b1.equity_scale == pytest.approx(0.5)


def test_no_dividend_below_minimum_ratio():
    m = StylizedInsurer(synthetic())
    b0 = m.initial()
    weak = InsurerState(
        available_capital=0.9 * b0.required_capital,
        required_capital=b0.required_capital,
        risks=b0.risks,
        operational=b0.operational,
        basic_capital=0.9 * b0.required_capital - b0.supplementary_capital,
        supplementary_capital=b0.supplementary_capital,
    )
    plain = m.step(weak, ZERO, Regime.NONE, NO_ACTION)
    cut = m.step(weak, ZERO, Regime.NONE, Action(dividend_cut=True))
    assert plain.available_capital == pytest.approx(cut.available_capital)


def test_action_cost_matches_capital_difference():
    m = StylizedInsurer(synthetic())
    b0 = m.initial()
    x = factor("loss_ratio", 0.03)
    a = Action(reinsurance_cession=0.2, dividend_cut=False)
    cost = m.action_cost(b0, x, a)
    plain = m.step(b0, x, Regime.NONE, NO_ACTION).basic_capital
    ceded = m.step(b0, x, Regime.NONE, a).basic_capital
    assert plain - ceded == pytest.approx(cost["cession"])
    assert cost["derisk"] == 0 and cost["sub_debt_carry"] == 0
