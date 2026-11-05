"""공시 복원 보험사 A~D 검사. 기준 시점 비율이 공시와 맞는지, 결정 D-010·D-011을 지키는지 본다."""
import numpy as np
import pytest

from interfaces.contracts import N_FACTORS, NO_ACTION, Regime
from model.archetypes import C_MEMBERS, D_TARGET_RATIO, build_all, load
from model.insurer import StylizedInsurer

ZERO = np.zeros(N_FACTORS)


@pytest.fixture(scope="module")
def archetypes():
    return build_all()


@pytest.fixture(scope="module")
def data():
    return load()


def test_single_firm_ratios_match_disclosure(archetypes, data):
    for key, firm in (("A", "삼성화재"), ("B", "DB손해보험")):
        b0 = StylizedInsurer(archetypes[key]).initial()
        disclosed = data[firm]["가용자본(경과조치 전)"] / data[firm]["요구자본(경과조치 전)"]
        assert b0.ratio == pytest.approx(disclosed, abs=1e-3)


def test_c_matches_summed_disclosure(archetypes, data):
    ac = sum(data[f]["가용자본(경과조치 전)"] for f in C_MEMBERS)
    rc = sum(data[f]["요구자본(경과조치 전)"] for f in C_MEMBERS)
    assert StylizedInsurer(archetypes["C"]).initial().ratio == pytest.approx(ac / rc, abs=1e-3)


def test_d_starts_at_target_ratio(archetypes):
    assert StylizedInsurer(archetypes["D"]).initial().ratio == pytest.approx(D_TARGET_RATIO)


def test_d_target_is_adjustable():
    d = build_all(target_ratio_d=1.40)["D"]
    assert StylizedInsurer(d).initial().ratio == pytest.approx(1.40)


def test_every_parameter_set_records_sources(archetypes):
    for p in archetypes.values():
        assert {"data", "corr", "members", "assumptions", "gaps"} <= set(p.sources)


def test_three_year_baseline_stays_solvent(archetypes):
    for p in archetypes.values():
        m = StylizedInsurer(p)
        b = m.initial()
        for _ in range(3):
            b = m.step(b, ZERO, Regime.NONE, NO_ACTION)
        assert b.ratio > 1.3


def test_new_business_hit_in_reform_year_then_repriced(archetypes):
    # 개편한 해에는 신계약 이익이 줄고, 이듬해에는 새 가정으로 재가격해 회복한다 (W40)
    m = StylizedInsurer(archetypes["B"])
    b0 = m.initial()
    hit = m.step(b0, ZERO, Regime.LOSS, NO_ACTION)
    assert m._new_business_factor(hit.liability_shift, b0.liability_shift) < 1.0
    nxt = m.step(hit, ZERO, Regime.NONE, NO_ACTION)
    assert m._new_business_factor(nxt.liability_shift, hit.liability_shift) == pytest.approx(1.0)


def test_lapse_reform_lowers_no_surrender_lapse(archetypes):
    # 2024-11 무·저해지 해지율 가이드라인 방향: 해지율을 낮추면 손실 (W21)
    for p in archetypes.values():
        assert p.regime_shift[Regime.LAPSE][2] < 1.0


def test_macro_contribution_signs(archetypes):
    from model.backtest import factor_change, macro_contribution

    start = {"ktb3": 2.5, "ktb10": 2.8, "aa3": 3.2, "kospi": 2500.0}
    rally = {"ktb3": 2.5, "ktb10": 2.8, "aa3": 3.2, "kospi": 3000.0}
    assert macro_contribution(archetypes["A"], factor_change(start, rally)) > 0


def test_b_rate_response_matches_disclosed_table(archetypes, data):
    # 정기경영공시 금리 민감도 표: +100bp 가용자본 226,544억, 요구자본 115,049억 (기준 235,232 / 115,099)
    m = StylizedInsurer(archetypes["B"])
    b0 = m.initial()
    x = np.zeros(N_FACTORS)
    x[0] = 0.01
    shocked, calm = m.step(b0, x, Regime.NONE, NO_ACTION), m.step(b0, ZERO, Regime.NONE, NO_ACTION)
    d_ac = shocked.available_capital - calm.available_capital
    d_rc = shocked.required_capital - calm.required_capital
    assert d_ac == pytest.approx((226544 - 235232) * 1e-4, rel=0.1)
    assert abs(d_rc) < 0.1


def _path(m, xs, actions=None):
    b = m.initial()
    for i, x in enumerate(xs):
        b = m.step(b, x, Regime.NONE, (actions or {}).get(i, NO_ACTION))
    return b


def _rate(v):
    x = np.zeros(N_FACTORS)
    x[0] = v
    return x


def test_rate_path_independence(archetypes):
    # 리뷰 1: 끝 금리가 같으면 경로가 달라도 거의 같은 결과. 이익이 자본 수준에 비례해 복리로 붙는 차이(≤1.5%p)만 허용
    for p in archetypes.values():
        m = StylizedInsurer(p)
        slow = _path(m, [_rate(0.01)] * 3)
        fast = _path(m, [_rate(0.03), ZERO, ZERO])
        assert slow.ratio == pytest.approx(fast.ratio, abs=0.015)
        round_trip = _path(m, [_rate(0.01), _rate(-0.01)])
        flat = _path(m, [ZERO, ZERO])
        assert round_trip.ratio == pytest.approx(flat.ratio, abs=0.015)


def test_no_free_derisking_without_equity(archetypes):
    # 리뷰 2: 주식을 다 판 뒤에는 축소 대응이 없고, 주가 충격이 시장위험을 줄이지 않는다
    m = StylizedInsurer(archetypes["B"])
    from interfaces.contracts import Action

    sold = m.step(m.initial(), ZERO, Regime.NONE, Action(derisk=1.0, dividend_cut=True))
    assert all(a.derisk == 0 for a in m.feasible_actions(sold, ZERO))
    crash = np.zeros(N_FACTORS)
    crash[3] = np.log(0.5)
    assert m.step(sold, crash, Regime.NONE, NO_ACTION).risks[2] == pytest.approx(
        m.step(sold, ZERO, Regime.NONE, NO_ACTION).risks[2]
    )


def test_strategic_floor_counts_shares_not_prices(archetypes):
    # 리뷰 3: 주가가 떨어져도 팔 수 있는 수량은 그대로다
    m = StylizedInsurer(archetypes["A"])
    crash = np.zeros(N_FACTORS)
    crash[3] = np.log(0.7)
    after = m.step(m.initial(), crash, Regime.NONE, NO_ACTION)
    assert any(a.derisk == pytest.approx(0.3) for a in m.feasible_actions(after, ZERO))


def test_plain_balance_sheet_with_negative_capital(archetypes):
    # 리뷰 5: 음수 가용자본을 일반 BalanceSheet로 넘겨도 그대로 복원된다
    from interfaces.contracts import BalanceSheet

    m = StylizedInsurer(archetypes["D"])
    rc0 = m.initial().required_capital
    _, _, basic, supp, _ = m._unpack(BalanceSheet(-5.0, rc0, 1))
    assert basic + supp == pytest.approx(-5.0)


def test_b_mass_lapse_close_to_disclosed_lapse_charge(archetypes, data):
    # 25% 대량해지의 가용자본 손실이 공시 해지위험액(세후)과 비슷해야 한다 (W47 검증).
    # 해지위험액은 요구자본을 고정한 손실이므로 가용자본 변화와 비교한다. 비율 효과는 보유계약 감소로 요구자본도 줄어 더 작다.
    p = archetypes["B"]
    m = StylizedInsurer(p)
    b0 = m.initial()
    x = np.zeros(N_FACTORS)
    x[5] = 0.25 / p.liability.first_year_lapse
    shocked, calm = m.step(b0, x, Regime.NONE, NO_ACTION), m.step(b0, ZERO, Regime.NONE, NO_ACTION)
    loss = shocked.available_capital - calm.available_capital
    implied = -data["DB손해보험"]["생명장기 하위위험-해지"] * 1e-4 * (1 - p.tax_rate)
    assert loss == pytest.approx(implied, rel=0.2)
    assert shocked.inforce == pytest.approx(0.75)


def test_rate_floor_and_calibration_flag(archetypes):
    m = StylizedInsurer(archetypes["B"])
    x = np.zeros(N_FACTORS)
    x[0] = -0.06
    deep = m.step(m.initial(), x, Regime.NONE, NO_ACTION)
    x[0] = -0.08
    deeper = m.step(m.initial(), x, Regime.NONE, NO_ACTION)
    assert deep.rate_floored and deep.outside_calibration
    assert deeper.ratio == pytest.approx(deep.ratio)
    assert not m.step(m.initial(), ZERO, Regime.NONE, NO_ACTION).outside_calibration


def test_multi_year_action_caps(archetypes):
    from model.insurer import simulate_path

    m = StylizedInsurer(archetypes["B"])
    b = m.initial()
    for _ in range(4):
        acts = m.feasible_actions(b, ZERO)
        b = max((m.step(b, ZERO, Regime.NONE, a) for a in acts), key=lambda s: s.ratio)
    assert b.retention >= 0.6 - 1e-9
    assert b.sub_debt_issued <= 0.10 * m.initial().available_capital + 1e-9
    assert b.equity_units >= 0.4 - 1e-9
    states = simulate_path(m, [ZERO, ZERO])
    assert len(states) == 3 and states[0].t == 0 and states[-1].t == 2
    assert states[1].risks.market == pytest.approx(states[1].risks[2])
