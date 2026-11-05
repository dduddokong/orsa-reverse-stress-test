"""공시 기반 감응도 모형: 가상 손보사 한 곳의 연 단위 K-ICS 이월.

- 가용자본(AC)은 위험요인 편차 x에 대한 감응도로 재평가하고, 기준 성장과 배당을 반영한다.
  기본자본과 보완자본을 나눠 이월하고, 보완자본은 요구자본의 50%까지만 인정한다(별표22 Ⅲ.2.마).
- 요구자본(RC)은 대분류 위험액을 x와 계리 체제 점프로 늘리거나 줄인 뒤 K-ICS 상관행렬로 합산한다.
  경과조치 인식비율(별표22 <표42>~<표44>)을 연도별로 적용하고, 법인세조정액은
  기본요구자본 × 평균세율(한도 있음)로 다시 계산한다(별표22 Ⅳ 제7장).
- 신계약이 들어와 보유계약 규모가 유지된다(D-009). 기준 전망에서 위험액·운영위험액은 rc_growth로 자란다.
- x는 그 해에 일어난 기준 전망 대비 변화로 해석한다. 효과는 상태를 통해 다음 해로 이어진다.

상태 운반: step()은 BalanceSheet의 하위 클래스 InsurerState를 돌려준다. 계약상 타입은 그대로이고,
대분류 위험액과 자본 계층을 함께 싣는다. 호출자가 일반 BalanceSheet를 넘기면 기준 시점 구성비로 복원한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from interfaces.contracts import (
    FACTORS,
    N_FACTORS,
    NO_ACTION,
    Action,
    BalanceSheet,
    Regime,
    check_factors,
)

from .kics import N_MAJOR, aggregate, check_corr, required_capital
from .liability import NO_SHIFT, LiabilityBlock, Shift

_RATE = FACTORS.index("rate_level")
_SLOPE = FACTORS.index("rate_slope")
_SPREAD = FACTORS.index("credit_spread")
_LOSS = FACTORS.index("loss_ratio")
_EQUITY = FACTORS.index("equity")
_INSURANCE_FACTORS = [FACTORS.index(f) for f in ("loss_ratio", "lapse", "expense_infl")]
_LAPSE = FACTORS.index("lapse")
_INFL = FACTORS.index("expense_infl")
_EOK = 1e-4  # 억원 → 조원
_INSURANCE = np.array([1.0, 1.0, 0.0, 0.0])  # 출재가 줄이는 대분류
_MARKET = np.array([0.0, 0.0, 1.0, 0.0])  # 위험자산 축소가 줄이는 대분류

SUPPLEMENTARY_LIMIT = 0.50  # 보완자본 인정 한도, 총요구자본 대비 (별표22 Ⅲ.2.마)


class RiskCharges(NamedTuple):
    """대분류 위험액 (조원, 경과조치 전). 순서는 kics.MAJOR_RISKS와 같다."""

    insurance_long: float
    insurance_general: float
    market: float
    credit: float


@dataclass(frozen=True)
class InsurerState(BalanceSheet):
    """BalanceSheet에 대분류 위험액(경과조치 전)과 자본 계층을 더한 상태."""

    risks: Tuple[float, ...] = ()  # RiskCharges
    operational: float = 0.0
    basic_capital: float = 0.0
    supplementary_capital: float = 0.0  # 한도 적용 전 합계
    supplementary_uncapped: float = 0.0  # 그중 50% 한도 밖 (해약환급금 부족분 재분류 등)
    equity_scale: float = 1.0  # 기준 대비 남은 주식 노출 (위험자산 축소와 주가 하락으로 줄어든다)
    retention: float = 1.0  # 기준 대비 보유 보험위험 비율 (출재로 줄어든다)
    liability_shift: Tuple[float, ...] = NO_SHIFT  # 누적 계리가정 변화 (liability.Shift 순서)
    rate_level: float = 0.0  # 기준 대비 누적 금리 변화 (자산 쪽 2차항을 경로와 무관하게 하려고 둔다)
    equity_units: float = 1.0  # 기준 대비 남은 주식 수량 (매도로만 줄고 주가에는 반응하지 않는다)
    inforce: float = 1.0  # 기준 대비 남은 보유계약 비율 (해지율 경험으로 줄거나 는다)
    sub_debt_issued: float = 0.0  # 기준 시점 이후 누적 후순위채 발행액 (조원)
    rate_floored: bool = False  # 금리 하한에 걸려 이번 해 금리 하락의 일부를 반영하지 않았는지

    @property
    def basic_ratio(self) -> float:
        """기본자본 K-ICS 비율 (2027년 50% 기준)."""
        return self.basic_capital / self.required_capital

    @property
    def outside_calibration(self) -> bool:
        """보정 범위 밖의 상태인지. 결과를 해석할 때 주의한다 (README '유효 범위')."""
        d_loss, lapse_mult = self.liability_shift[0], self.liability_shift[1]
        return (
            self.rate_floored
            or abs(self.rate_level) > 0.02
            or abs(d_loss) > 0.10
            or not 0.5 <= lapse_mult <= 2.0
            or not 0.4 <= self.liability_shift[5] <= 1.0  # 무·저해지 해지율 개편이 두 번 이상 겹친 상태
            or self.equity_scale > 3.0
        )


@dataclass(frozen=True)
class ActionTerms:
    """경영대응의 효과·비용·실행 조건. 후보 집합은 DECISIONS에서 확정한다."""

    cession_levels: Tuple[float, ...] = (0.0, 0.2)
    cession_relief: float = 0.3  # 출재율 1당 보험위험액 감소 비율
    cession_cost: float = 0.02  # 출재율 1당 AC 대비 비용 (평시)
    cession_cost_per_loss: float = 0.1  # 손해율 +1(=100%p)당 추가 비용 기울기
    derisk_levels: Tuple[float, ...] = (0.0, 0.3)
    derisk_relief: float = 0.2  # 축소 비중 1당 시장위험액 감소 비율
    derisk_cost: float = 0.0  # 축소 비중 1당 평시 매각 비용 (AC 대비)
    min_equity_scale: float = 0.0  # 팔 수 없는 전략적 보유분 (기준 수량 대비 하한, 주가와 무관)
    derisk_cost_per_spread: float = 1.0  # 스프레드 +1당 매각 손실 (AC 대비, 축소 비중 1당)
    derisk_cost_per_equity_drop: float = 0.0  # 주가 하락률 1당 추가 매각 손실 (AC 대비, 축소 비중 1당)
    cession_credit: float = 0.0  # 출재율 1당 신용위험액 증가 (조원, 재보험자산 거래상대방)
    sub_debt_levels: Tuple[float, ...] = (0.0, 0.05)
    sub_debt_carry: float = 0.0  # 발행액 1당 첫해 세후 음의 캐리 (쿠폰 − 운용수익률)
    sub_debt_max_spread: float = 0.015  # 스프레드 편차가 이보다 크면 발행 불가
    sub_debt_min_ratio: float = 1.0  # 직전 K-ICS 비율이 이보다 낮으면 발행 불가
    # 여러 해에 걸친 한도 (B 차선 시험: 대응이 해마다 쌓여 내부 최대화가 늘 '전부'가 됨) [결정 필요]
    min_retention: float = 0.0  # 누적 출재 후 보유율 하한
    max_sub_debt_total: float = 1e9  # 누적 후순위채 발행 한도 (기준 가용자본 대비)


@dataclass(frozen=True)
class Transitional:
    """경과조치. phased_share는 대분류 위험액 중 인식비율이 적용되는 몫."""

    phased_share: np.ndarray = field(default_factory=lambda: np.zeros(N_MAJOR))
    insurance_schedule: Mapping[int, float] = field(default_factory=dict)  # <표42>
    market_schedule: Mapping[int, float] = field(default_factory=dict)  # <표43>·<표44>

    def recognition(self, year: int) -> np.ndarray:
        """연도별 대분류 위험액 인식 배수 (1이면 전액 인식)."""
        ins = _lookup(self.insurance_schedule, year)
        mkt = _lookup(self.market_schedule, year)
        rate = np.array([ins, ins, mkt, 1.0])
        return 1.0 - np.asarray(self.phased_share) * (1.0 - rate)


def _lookup(schedule: Mapping[int, float], year: int) -> float:
    if not schedule:
        return 1.0
    past = [y for y in schedule if y <= year]
    return float(schedule[max(past)]) if past else float(schedule[min(schedule)])


@dataclass(frozen=True)
class InsurerParams:
    """가상 보험사 한 곳의 모수. 금액 단위는 조원, 비율은 소수."""

    name: str
    base_year: int  # 기준 시점 연도. 1년 이월 후는 base_year + 1
    basic_capital: float
    supplementary_capital: float  # 한도 적용 전 합계
    risks: np.ndarray  # 대분류 위험액, 경과조치 전 (MAJOR_RISKS 순서)
    corr: np.ndarray  # 대분류 상관행렬
    operational: float
    tax_rate: float  # 평균세율
    tax_cap: float  # 법인세조정액 한도 (조원)
    ac_growth: float  # 기준 전망 AC 증가율 (배당 전)
    rc_growth: float  # 기준 전망 위험액·운영위험액 증가율
    dividend_payout: float  # AC 대비 배당
    ac_sens: np.ndarray  # 위험요인 1단위당 AC 변화 (AC 대비), 길이 N_FACTORS
    ac_rate_convexity: float  # 금리 변화 제곱항 (AC 대비, 음수면 양방향 손실)
    risk_sens: np.ndarray  # 위험요인 1단위당 대분류 위험액 상대 변화, (N_MAJOR, N_FACTORS)
    regime_ac_hit: Mapping[Regime, float]  # 체제 점프의 AC 감소 (AC 대비)
    regime_risk_rise: Mapping[Regime, np.ndarray]  # 체제 점프의 대분류 위험액 상대 증가
    uncapped_supplementary: float = 0.0  # 보완자본 중 50% 한도 밖 몫 (별표22 Ⅲ.2.다.(3), Ⅲ.2.마)
    scheduled_ac_hit: Mapping[int, float] = field(default_factory=dict)  # 연도별 예정 제도변화의 AC 감소 (AC 대비)
    liability: Optional[LiabilityBlock] = None  # 1단계 현금흐름 부채. 있으면 손해율·해지율·사업비 경로를 대신한다
    # 체제 점프 = (손해율 +, 전체 해지율 ×, 무·저해지 해지율 ×)
    regime_shift: Mapping[Regime, Tuple[float, float, float]] = field(default_factory=dict)
    # 신계약 (W5·W22): 규모 유지에서 신계약 CSM(세후)은 상각을 메워 CSM을 일정하게 한다. 신계약 이익이 f배가 되면
    # CSM이 (f−1)×신계약 CSM만큼 덜 쌓여 가용자본이 그만큼 준다. 순이익 기반 ac_growth의 일부가 아니므로
    # new_business_growth가 ac_growth보다 클 수 있다 (D처럼 자본이 얇은 회사).
    new_business: Optional[LiabilityBlock] = None
    new_business_growth: float = 0.0
    va_pass_through: float = 0.0  # 신용스프레드 1 상승당 부채 할인율 상승 (변동성조정, W16)
    # 손해율·해지율 위험요인 = 경험 변동. 그 해 손익에 전부 반영되고, 장래 가정에는 신뢰도만큼만 반영된다
    # (해지율은 한 해 ±25% 상한).
    # 체제 점프(감독 개편)는 가정 자체를 바꾸므로 전부 반영된다. 1이면 경험이 곧 가정 변화 (리뷰 ② 요인·점프 중복).
    experience_credibility: float = 1.0
    llp_schedule: Mapping[int, int] = field(default_factory=dict)  # 연도별 최종관찰만기 (W12)
    other_required: float = 0.0  # 기타요구자본 (조원, 고정). 여러 회사를 통합할 때 회사 간 분산효과 되돌림에도 쓴다
    min_rate_level: float = -1e9  # 누적 금리 변화 하한 (기준 시장 금리 + 하한 = 0.5%)
    transitional: Transitional = Transitional()
    actions: ActionTerms = ActionTerms()
    sources: Mapping[str, str] = field(default_factory=dict)  # 모수별 출처·주의사항

    def __post_init__(self) -> None:
        check_corr(self.corr)
        if np.asarray(self.risks).shape != (N_MAJOR,):
            raise ValueError("risks must have one entry per major risk")
        if np.asarray(self.ac_sens).shape != (N_FACTORS,):
            raise ValueError("ac_sens must have length N_FACTORS")
        if np.asarray(self.risk_sens).shape != (N_MAJOR, N_FACTORS):
            raise ValueError("risk_sens must have shape (N_MAJOR, N_FACTORS)")
        if set(self.regime_ac_hit) != set(Regime) or set(self.regime_risk_rise) != set(Regime):
            raise ValueError("regime tables must cover every Regime")


class StylizedInsurer:
    """interfaces.contracts.KicsModel 구현."""

    def __init__(self, params: InsurerParams) -> None:
        self.p = params
        self._liability_cache: Dict[Shift, Tuple[float, float]] = {}
        self._nb_cache: Dict[Shift, float] = {}
        self._rc0 = self.initial().required_capital

    def _new_business_factor(self, shift: Shift, shift_prev: Shift, llp: Optional[int] = None) -> float:
        """올해 판 신계약의 장래이익 ÷ 작년 가정에서의 장래이익. 손실이면 판매를 멈춘다고 보고 0.

        보험료는 작년 가정으로 정해져 있다가 이듬해 새 가정으로 다시 정한다(1년 시차 재가격, W40).
        그래서 가정이 나빠진 해에만 신계약 이익이 줄고, 다음 해에는 회복한다. 매년 나빠지면 매년 준다.
        """
        nb = self.p.new_business
        if nb is None:
            return 1.0
        # 금리·기울기는 빼고 계리가정 변화만 본다. 금리 효과는 공시 재평가 민감도에 이미 있고,
        # 장래이익이 작은 신계약 블록에서는 금리 비율 효과가 과장된다 (11회차 점검에서 A의 부호 역전).
        shift = shift[:3] + shift_prev[3:5] + shift[5:]
        key = tuple(round(v, 12) for v in shift) + tuple(round(v, 12) for v in shift_prev) + (llp,)
        if key not in self._nb_cache:
            # 신계약 장래이익은 −BEL로 본다 (위험마진 제외, 비율만 쓰므로 계산을 줄인다)
            base = -nb.best_estimate(tuple(round(v, 12) for v in shift_prev), llp)
            now = -nb.best_estimate(tuple(round(v, 12) for v in shift), llp)
            self._nb_cache[key] = max(now / base, 0.0) if base > 0 else 1.0
        return self._nb_cache[key]

    def _llp(self, t: int) -> Optional[int]:
        """t년 뒤 최종관찰만기 (예정 일정, W12). 일정이 없으면 계리가정의 값."""
        sched = self.p.llp_schedule
        if not sched:
            return None
        year = self.p.base_year + t
        past = [y for y in sched if y <= year]
        return int(sched[max(past)] if past else sched[min(sched)])

    def _liability(self, shift: Shift, t: int = 0) -> Tuple[float, float]:
        """현금흐름 부채의 (장래이익, 장기위험액), 조원. 같은 가정 조합은 다시 계산하지 않는다."""
        llp = self._llp(t)
        key = tuple(round(v, 12) for v in shift) + (llp,)
        if key not in self._liability_cache:
            of, long = self.p.liability.evaluate(tuple(round(v, 12) for v in shift), llp)
            self._liability_cache[key] = (of * _EOK, long * _EOK)
        return self._liability_cache[key]

    # 상태 조립 -----------------------------------------------------------
    def _state(
        self,
        risks: np.ndarray,
        operational: float,
        basic: float,
        supplementary: float,
        uncapped: float,
        t: int,
        equity_scale: float = 1.0,
        retention: float = 1.0,
        liability_shift: Shift = NO_SHIFT,
        rate_level: float = 0.0,
        equity_units: float = 1.0,
        inforce: float = 1.0,
        sub_debt_issued: float = 0.0,
        rate_floored: bool = False,
    ) -> InsurerState:
        p = self.p
        recognised = risks * p.transitional.recognition(p.base_year + t)
        rc = required_capital(recognised, p.corr, operational, p.tax_rate, p.tax_cap) + p.other_required
        rc = max(rc, 1e-9)
        capped = max(supplementary - uncapped, 0.0)
        ac = basic + uncapped + min(capped, SUPPLEMENTARY_LIMIT * rc)
        return InsurerState(
            available_capital=ac,
            required_capital=rc,
            t=t,
            risks=RiskCharges(*(float(r) for r in risks)),
            operational=operational,
            basic_capital=basic,
            supplementary_capital=supplementary,
            supplementary_uncapped=uncapped,
            equity_scale=equity_scale,
            retention=retention,
            liability_shift=liability_shift,
            rate_level=rate_level,
            equity_units=equity_units,
            inforce=inforce,
            sub_debt_issued=sub_debt_issued,
            rate_floored=rate_floored,
        )

    def _unpack(self, prev: BalanceSheet) -> Tuple[np.ndarray, float, float, float, float]:
        """prev에서 위험액·운영위험액·기본자본·보완자본(합계, 한도 밖)을 꺼낸다."""
        if isinstance(prev, InsurerState) and prev.risks:
            return (
                np.asarray(prev.risks),
                prev.operational,
                prev.basic_capital,
                prev.supplementary_capital,
                prev.supplementary_uncapped,
            )
        # 일반 BalanceSheet: 기준 시점 구성비를 RC·AC 크기에 맞춰 늘린다.
        # 한도를 넘어 인정되지 않던 보완자본은 복원하지 못한다.
        b0 = self.initial()
        k = prev.required_capital / b0.required_capital
        # 가용자본이 0 이하이면 보완자본을 복원할 근거가 없으므로 전부 기본자본으로 둔다.
        scale = max(prev.available_capital, 0.0) / b0.available_capital
        supp = (b0.available_capital - b0.basic_capital) * scale
        uncapped = b0.supplementary_uncapped * scale
        return (
            np.asarray(b0.risks) * k,
            b0.operational * k,
            prev.available_capital - supp,
            supp,
            uncapped,
        )

    # 계약 ----------------------------------------------------------------
    def initial(self) -> InsurerState:
        p = self.p
        return self._state(
            np.asarray(p.risks, dtype=float),
            p.operational,
            p.basic_capital,
            p.supplementary_capital,
            p.uncapped_supplementary,
            0,
        )

    def step(
        self,
        prev: BalanceSheet,
        x: np.ndarray,
        regime: Regime,
        action: Action,
    ) -> InsurerState:
        x = check_factors(x)
        # 금리 하한: 누적 금리가 min_rate_level 아래로 내려가는 몫은 반영하지 않는다 (B 차선 시험: 음의 금리 영역에서
        # 부채가 폭증해 최적화가 그쪽을 붕괴 경로로 고름). 하한에 걸린 해는 상태에 표시한다.
        rate_prev = getattr(prev, "rate_level", 0.0)
        floor_hit = rate_prev + x[_RATE] < self.p.min_rate_level
        if floor_hit:
            x = x.copy()
            x[_RATE] = self.p.min_rate_level - rate_prev
        # 주가 요인은 로그수익률이다. 재평가와 위험액에는 실제 가격 변화율 e^x − 1을 쓴다 (W39).
        x_lin = x.copy()
        x_lin[_EQUITY] = float(np.expm1(x[_EQUITY]))
        p, terms = self.p, self.p.actions
        risks, operational, basic, supplementary, uncapped = self._unpack(prev)
        equity_scale = getattr(prev, "equity_scale", 1.0)
        equity_units = getattr(prev, "equity_units", 1.0)
        rate_level = getattr(prev, "rate_level", 0.0)
        retention = getattr(prev, "retention", 1.0)
        inforce = getattr(prev, "inforce", 1.0)
        issued = getattr(prev, "sub_debt_issued", 0.0)
        ac = prev.available_capital
        ac_pos = max(ac, 0.0)  # 이익·배당·후순위채는 자본이 음수면 0으로 본다 (리뷰 6)

        # 지난해까지의 대응으로 줄어든 노출만큼 감응도를 줄인다 (W19·W20).
        sens = np.array(p.ac_sens, dtype=float)
        sens[_EQUITY] *= equity_scale
        sens[_INSURANCE_FACTORS] *= retention

        shift_prev = getattr(prev, "liability_shift", NO_SHIFT)
        shift = shift_prev
        if p.liability is not None:
            j_loss, j_lapse, j_nosv = p.regime_shift.get(regime, (0.0, 1.0, 1.0))
            shift = (
                shift_prev[0] + p.experience_credibility * x[_LOSS] + j_loss,
                # 한 해 경험이 해지율 가정을 바꾸는 폭은 ±25%로 제한한다 (대량해지 한 해가 영구 가정이 되지 않게) [가정]
                shift_prev[1] * (1.0 + float(np.clip(p.experience_credibility * x[_LAPSE], -0.25, 0.25))) * j_lapse,
                shift_prev[2] + x[_INFL],
                shift_prev[3] + x[_RATE] + p.va_pass_through * x[_SPREAD],
                shift_prev[4] + x[_SLOPE],
                shift_prev[5] * j_nosv,
            )
        nb_growth = p.new_business_growth * self._new_business_factor(shift, shift_prev, self._llp(prev.t + 1))

        # 가용자본: 손익과 재평가는 기본자본으로, 후순위채는 보완자본으로 간다.
        # 이익·배당은 직전 AC에 비례하고, 자산·부채 재평가와 대응 비용은 자산·부채 크기(기준 AC로 잰 값)에
        # 비례한다. 자본이 줄어도 자산 손실이 같이 줄지 않게 하려는 것이다 (W34).
        ac0 = p.basic_capital + p.uncapped_supplementary + min(
            p.supplementary_capital - p.uncapped_supplementary, SUPPLEMENTARY_LIMIT * self._rc0
        )
        new_rate = rate_level + x[_RATE]
        d_basic = ac_pos * (p.ac_growth - p.new_business_growth + nb_growth) + ac0 * (
            sens @ x_lin
            # 누적 금리 수준의 2차식 차분: 같은 끝 금리면 경로와 무관하다 (리뷰 1)
            + p.ac_rate_convexity * (new_rate**2 - rate_level**2)
            - p.regime_ac_hit[regime]
            - p.scheduled_ac_hit.get(p.base_year + prev.t + 1, 0.0)
        )
        # 직전 비율이 100% 미만이면 적기시정조치로 배당이 막힌다고 본다 (W33, 감독규정 조문 미확인).
        if not action.dividend_cut and prev.ratio >= 1.0:
            d_basic -= ac_pos * p.dividend_payout
        costs = self._action_costs(prev, x, x_lin, action)
        d_basic -= costs["cession"] + costs["derisk"] + costs["sub_debt_carry"]
        d_supp = ac_pos * action.sub_debt

        # 1단계 부채: 손해율·해지율·사업비·할인율 변화와 체제 점프를 가정 변화로 넣고 BEL을 다시 계산한다.
        # 금리의 자산 쪽 효과는 ac_sens에 남아 있다 (공시 순민감도 − 부채 몫).
        d_long = 0.0
        new_inforce = inforce
        if p.liability is not None:
            w = retention * inforce  # 출재하고 해지로 줄어든 뒤 남은 보유계약 몫
            # 직전 해의 곡선에서 이번 해 곡선으로: 최종관찰만기 연장 효과도 여기서 생긴다.
            of0, long0 = self._liability(shift_prev, prev.t)
            of1, long1 = self._liability(shift, prev.t + 1)
            d_basic += (of1 - of0) * (1.0 - p.tax_rate) * w
            d_long = (long1 - long0) * w
            # 손해율은 위험보험료 대비이므로 당해 보험금 변화 = 첫해 위험보험료 × x (리뷰3-1)
            d_basic -= p.liability.first_year_risk_premium * _EOK * x[_LOSS] * (1.0 - p.tax_rate) * w
            # 현재 해지율 가정(누적 배수)을 기준으로 한 초과 해지 비율. 해지된 몫은 보유계약에서 빠져
            # 다음 해 재평가·장기위험액도 그만큼 준다 (리뷰3-3, 3-5).
            excess = min(max(p.liability.first_year_lapse * shift_prev[1] * x[_LAPSE], -0.5), 0.9)
            d_basic -= excess * of1 * (1.0 - p.tax_rate) * w
            d_long -= excess * long1 * w
            new_inforce = inforce * (1.0 - excess)

        # 요구자본: 규모 성장 → 위험요인·체제 점프 → 경영대응.
        # 주식위험액 탄력성과 축소 경감은 남은 주식 노출에 비례한다 (리뷰 2).
        rsens = np.array(p.risk_sens, dtype=float)
        rsens[:, _EQUITY] *= equity_scale
        mult = 1.0 + p.rc_growth + rsens @ x_lin + np.asarray(p.regime_risk_rise[regime])
        mult = mult * (1.0 - action.reinsurance_cession * terms.cession_relief * _INSURANCE)
        mult = mult * (1.0 - action.derisk * terms.derisk_relief * equity_scale * _MARKET)
        new_risks = risks * mult
        new_risks[0] += d_long
        new_risks[3] += action.reinsurance_cession * retention * new_inforce * terms.cession_credit  # 리뷰3-2, 4-1, 5-6
        new_risks = np.maximum(new_risks, 0.0)

        return self._state(
            new_risks,
            operational * (1.0 + p.rc_growth),
            basic + d_basic,
            supplementary + d_supp,
            uncapped,
            prev.t + 1,
            # 남은 주식 노출은 매도분만큼 줄고 주가 변화만큼 늘거나 준다 (W28)
            equity_scale * (1.0 - action.derisk) * float(np.exp(x[_EQUITY])),
            retention * (1.0 - action.reinsurance_cession),
            shift,
            new_rate,
            equity_units * (1.0 - action.derisk),
            new_inforce,
            issued + d_supp,
            floor_hit,
        )

    def _action_costs(
        self, prev: BalanceSheet, x: np.ndarray, x_lin: np.ndarray, action: Action
    ) -> Dict[str, float]:
        """대응별 가용자본 비용(조원, 세후, 양수가 비용). step()과 action_cost()가 같이 쓴다."""
        p, terms = self.p, self.p.actions
        retention = getattr(prev, "retention", 1.0)
        inforce = getattr(prev, "inforce", 1.0)
        equity_scale = getattr(prev, "equity_scale", 1.0)
        ac0 = self.initial().available_capital
        ac_pos = max(prev.available_capital, 0.0)
        return {
            "cession": ac0 * action.reinsurance_cession * retention * inforce * (
                terms.cession_cost + terms.cession_cost_per_loss * max(x[_LOSS], 0.0)
            ),
            # 연말에 떨어진 가격으로 판다 (리뷰3-4)
            "derisk": ac0 * action.derisk * equity_scale * (
                terms.derisk_cost
                + terms.derisk_cost_per_spread * max(x[_SPREAD], 0.0)
                + terms.derisk_cost_per_equity_drop * max(-x_lin[_EQUITY], 0.0)
            ) * (1.0 + x_lin[_EQUITY]),
            "sub_debt_carry": ac_pos * action.sub_debt * terms.sub_debt_carry,
            # 주주에게 가는 비용(가용자본에는 이득). 비율 100% 미만이면 어차피 못 주므로 0.
            "dividend_forgone": ac_pos * p.dividend_payout if action.dividend_cut and prev.ratio >= 1.0 else 0.0,
        }

    def action_cost(self, prev: BalanceSheet, x: np.ndarray, action: Action) -> Dict[str, float]:
        """대응 비용 내역 (조원). B 차선의 바깥 문제(편익 − λ·비용)와 H2 분석에 쓴다.

        cession·derisk·sub_debt_carry는 가용자본에서 빠지는 비용이고, dividend_forgone은 주주가 못 받는 배당이다.
        """
        x = check_factors(x)
        x_lin = x.copy()
        x_lin[_EQUITY] = float(np.expm1(x[_EQUITY]))
        return self._action_costs(prev, x, x_lin, action)

    def feasible_actions(self, prev: BalanceSheet, x: np.ndarray) -> List[Action]:
        x = check_factors(x)
        terms = self.p.actions
        equity_scale = getattr(prev, "equity_scale", 1.0)
        equity_units = getattr(prev, "equity_units", 1.0)
        retention = getattr(prev, "retention", 1.0)
        issued = getattr(prev, "sub_debt_issued", 0.0)
        ac0 = self.initial().available_capital
        sub_debt_open = (
            x[_SPREAD] <= terms.sub_debt_max_spread
            and prev.ratio >= terms.sub_debt_min_ratio
            and self._supplementary_room(prev) > 0
        )
        actions = [NO_ACTION]
        for cession in terms.cession_levels:
            for derisk in terms.derisk_levels:
                for sub_debt in terms.sub_debt_levels:
                    if sub_debt > 0 and (
                        not sub_debt_open
                        or issued + sub_debt * max(prev.available_capital, 0.0) > terms.max_sub_debt_total * ac0 + 1e-12
                    ):
                        continue
                    if cession > 0 and retention * (1.0 - cession) < terms.min_retention - 1e-12:
                        continue
                    if derisk > 0 and (
                        equity_scale < 1e-6
                        or equity_units * (1.0 - derisk) < terms.min_equity_scale - 1e-12
                    ):
                        continue
                    a = Action(cession, derisk, sub_debt, dividend_cut=True)
                    if a != NO_ACTION:
                        actions.append(a)
        return actions

    def _supplementary_room(self, prev: BalanceSheet) -> float:
        """보완자본 한도까지 남은 여유 (조원)."""
        _, _, _, supplementary, uncapped = self._unpack(prev)
        return SUPPLEMENTARY_LIMIT * prev.required_capital - (supplementary - uncapped)


def simulate_path(
    model: "StylizedInsurer",
    xs: Sequence[np.ndarray],
    regimes: Optional[Sequence[Regime]] = None,
    actions: Optional[Sequence[Action]] = None,
) -> List[InsurerState]:
    """기준 시점부터 해마다 step()을 이어 부른 상태 목록 (길이 len(xs) + 1, 첫 원소는 기준 시점).

    상태(InsurerState)를 그대로 다음 해로 넘기므로 B 차선이 직접 이어 붙일 필요가 없다.
    """
    regimes = list(regimes) if regimes is not None else [Regime.NONE] * len(xs)
    actions = list(actions) if actions is not None else [NO_ACTION] * len(xs)
    states = [model.initial()]
    for x, regime, action in zip(xs, regimes, actions):
        states.append(model.step(states[-1], x, regime, action))
    return states


def describe(params: InsurerParams, state: Optional[InsurerState] = None) -> Dict[str, float]:
    """상태 요약: AC, RC, 비율, 기본자본비율, 분산효과."""
    m = StylizedInsurer(params)
    s = state if state is not None else m.initial()
    recognised = np.asarray(s.risks) * params.transitional.recognition(params.base_year + s.t)
    undiversified = float(np.sum(recognised)) + s.operational
    diversified = aggregate(recognised, params.corr) + s.operational
    return {
        "available_capital": s.available_capital,
        "required_capital": s.required_capital,
        "ratio": s.ratio,
        "basic_ratio": s.basic_ratio,
        "diversification": undiversified - diversified,
    }
