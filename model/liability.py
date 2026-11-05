"""1단계: 장기 인보험(건강보험) 모델포인트 현금흐름과 K-ICS 장기손해 하위위험.

모델포인트 하나는 남은 보험기간·납입기간이 같은 계약 묶음이다. 연 단위로 유지자 수를 굴리며
보험료(유입), 보험금·사업비·해약환급금(유출)을 만들고 현행추정부채(BEL)를 계산한다.
BEL이 음수면 장래이익이 있는 계약이다. K-ICS에서 그 이익(−BEL − 위험마진)은 가용자본에 들어간다.

하위위험은 별표22 Ⅳ 제2장 충격을 BEL 재계산으로 적용한다 (docs/KICS_표준모형_정리.md 3절).
- 장해·질병: 위험률 +13%(정액)/+10%(실손) → blended_disease_shock
- 해지: 해지율 ×1.35, ×0.65, 대량해지(보장성 25%) 중 최대 손실
- 사업비: 사업비 +10%와 인플레이션 +1%p 동시
위험마진은 장기위험액 ÷ Z99.5% × Z85% (별표22 Ⅱ.3-3, 정규분포 백분위 방식).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Z995 = 2.5758293035489004
Z85 = 1.0364333894937898
RM_FACTOR = Z85 / Z995

# 별표22 <표6>에서 사망·장해질병·해지·사업비 4개
CORR_LONG4 = np.array(
    [
        [1.00, 0.25, 0.00, 0.25],
        [0.25, 1.00, 0.00, 0.50],
        [0.00, 0.00, 1.00, 0.50],
        [0.25, 0.50, 0.50, 1.00],
    ]
)
SUB_RISKS = ("death", "disease", "lapse", "expense")


@dataclass(frozen=True)
class Assumptions:
    """계리가정. 손해율·해지율·사업비는 체제 점프와 위험요인이 바꾸는 대상이다."""

    loss_ratio: float = 0.85  # 예정위험보험료 대비 실제 보험금
    lapse_pay: float = 0.06  # 납입 중 연 해지율
    lapse_paid: float = 0.008  # 완납 후 연 해지율 (2024-11 가이드라인 해외통계 0.8%)
    expense_ratio: float = 0.08  # 납입 중 유지비 (보험료 대비)
    expense_paid: float = 0.02  # 완납 후 유지비 (최초 보험료 대비)
    inflation: float = 0.02  # 사업비 인플레이션
    discount: float = 0.035  # 최종관찰만기까지의 시장 할인율 (수준)
    ltfr: float = 0.043  # 장기선도금리 (금감원 2027 할인율 안내 4.30%)
    llp: int = 23  # 최종관찰만기(년). '26·'27 23년, '28·'29 24년, '35년 30년
    convergence: float = 0.13  # 최종관찰만기 뒤 수렴 속도. 금감원 Smith-Wilson α 0.130 (학계 발표 인용, 감독 원문 미확인)
    death_rate: float = 0.003  # 사망 탈퇴율


@dataclass(frozen=True)
class ModelPoint:
    """남은 보험기간·납입기간이 같은 계약 묶음. 금액 단위는 억원(유지자 전체)."""

    name: str
    term: int  # 남은 보험기간(년)
    pay_years: int  # 남은 납입기간(년)
    premium: float  # 첫해 보험료 합계
    risk_share: float  # 보험료 중 예정위험보험료 비중
    claim_aging: float = 0.03  # 연령 증가에 따른 위험률 상승 (연)
    surrender_value: float = 0.0  # 해지 1건당 환급금 (보험료 대비). 무·저해지는 0에 가깝다
    death_benefit_share: float = 0.1  # 보험금 중 사망보험금 비중 (사망위험 충격 대상)
    no_surrender: bool = False  # 무·저해지 상품 (해지율 개편 대상)


def cash_flows(mp: ModelPoint, a: Assumptions) -> Dict[str, np.ndarray]:
    """연도별 유지자 비율과 현금흐름 (해당 연도 중간 발생으로 본다)."""
    t = np.arange(mp.term)
    paying = t < mp.pay_years
    lapse = np.where(paying, a.lapse_pay, a.lapse_paid)
    survive = np.concatenate([[1.0], np.cumprod(1.0 - lapse - a.death_rate)[:-1]])
    premium = np.where(paying, mp.premium, 0.0) * survive
    expected_claims = mp.premium * mp.risk_share * (1.0 + mp.claim_aging) ** t
    claims = a.loss_ratio * expected_claims * survive
    expense = np.where(
        paying, a.expense_ratio * mp.premium, a.expense_paid * mp.premium
    ) * (1.0 + a.inflation) ** t * survive
    surrender = lapse * survive * mp.surrender_value * mp.premium
    return {
        "survive": survive,
        "premium": premium,
        "claims": claims,
        "expense": expense,
        "surrender": surrender,
    }


def bel(mp: ModelPoint, a: Assumptions) -> float:
    """현행추정부채 = 유출 현가 − 유입 현가."""
    cf = cash_flows(mp, a)
    v = discount_factors(mp.term, np.array([a.discount]), 0.0, a.llp, a.ltfr, a.convergence)[0]
    net = cf["claims"] + cf["expense"] + cf["surrender"] - cf["premium"]
    return float(net @ v)


def pv_outflows(mps: Sequence[ModelPoint], a: Assumptions, d_rate: float = 0.0) -> float:
    """부채 유출(보험금·사업비·환급금)의 현가 (억원)."""
    total = 0.0
    for mp in mps:
        cf = cash_flows(mp, a)
        v = discount_factors(mp.term, np.array([a.discount + d_rate]), 0.0, a.llp, a.ltfr, a.convergence)[0]
        total += float((cf["claims"] + cf["expense"] + cf["surrender"]) @ v)
    return total


def outflow_duration(mps: Sequence[ModelPoint], a: Assumptions, h: float = 0.01) -> float:
    """부채 유출의 실효 듀레이션 (년). 공시 '부채 듀레이션'의 대용.

    금융위 2025-10-17 보도자료 p.4: 듀레이션은 유효듀레이션이며 "시장관찰구간 금리가 ±1%p 변동할 경우" 건전성 회계상
    부채가치의 변동비율이다. 그래서 시장 구간 금리를 ±1%p 움직인 현가 변화율로 계산한다. 최종관찰만기 뒤 곡선은
    LTFR로 수렴하므로 충격이 장기 구간에서 약해진다. 유출만 쓰는 것(보험료 유입 제외)은 정의 미확인 부분이다.
    """
    up, dn = pv_outflows(mps, a, h), pv_outflows(mps, a, -h)
    return (dn - up) / (2.0 * h * pv_outflows(mps, a))


def portfolio_bel(mps: Sequence[ModelPoint], a: Assumptions) -> float:
    return sum(bel(mp, a) for mp in mps)


def _scaled(mps: Sequence[ModelPoint], **changes: float) -> list:
    return [replace(mp, **changes) for mp in mps]


def sub_risks(
    mps: Sequence[ModelPoint],
    a: Assumptions,
    blended_disease_shock: float = 0.115,
    mass_lapse: float = 0.25,
) -> Dict[str, float]:
    """K-ICS 장기손해 하위위험액 (억원). 손실이 음수면 0."""
    base_each = [bel(mp, a) for mp in mps]
    base = sum(base_each)

    def loss(new_bel: float) -> float:
        return max(new_bel - base, 0.0)

    # 사망: 사망보험금 비중만큼의 위험률 +12.5% (근사)
    death = sum(
        max(bel(replace(mp, risk_share=mp.risk_share * (1 + 0.125 * mp.death_benefit_share)), a) - b0, 0.0)
        for mp, b0 in zip(mps, base_each)
    )
    disease = loss(portfolio_bel(mps, replace(a, loss_ratio=a.loss_ratio * (1 + blended_disease_shock))))
    up = loss(portfolio_bel(mps, replace(a, lapse_pay=a.lapse_pay * 1.35, lapse_paid=a.lapse_paid * 1.35)))
    down = loss(portfolio_bel(mps, replace(a, lapse_pay=a.lapse_pay * 0.65, lapse_paid=a.lapse_paid * 0.65)))
    # 대량해지: 보장성 25%가 즉시 해지. 남은 계약의 BEL은 75%로 줄고 환급금을 지급한다.
    mass = max(
        sum(mass_lapse * (-b0) + mass_lapse * mp.surrender_value * mp.premium for mp, b0 in zip(mps, base_each)),
        0.0,
    )
    lapse = max(up, down, mass)
    expense = loss(
        portfolio_bel(
            mps,
            replace(
                a,
                expense_ratio=a.expense_ratio * 1.1,
                expense_paid=a.expense_paid * 1.1,
                inflation=a.inflation + 0.01,
            ),
        )
    )
    return {"death": death, "disease": disease, "lapse": lapse, "expense": expense}


def long_risk(subs: Dict[str, float]) -> float:
    """하위위험을 <표6> 상관으로 합산한 장기손해보험위험액 (대재해 제외)."""
    s = np.array([subs[k] for k in SUB_RISKS])
    return float(np.sqrt(s @ CORR_LONG4 @ s))


def risk_margin(subs: Dict[str, float]) -> float:
    """K-ICS 위험마진 = 장기위험액(대재해 제외) ÷ Z99.5% × Z85%."""
    return long_risk(subs) * RM_FACTOR


def own_funds_contribution(mps: Sequence[ModelPoint], a: Assumptions) -> float:
    """보유계약이 가용자본에 더하는 몫(세전) = −(BEL + 위험마진)."""
    subs = sub_risks(mps, a)
    return -(portfolio_bel(mps, a) + risk_margin(subs))


# (손해율 +, 해지율 ×, 인플레이션 +, 10년 금리 +, 10년−3년 기울기 +, 무·저해지 해지율 ×)
Shift = Tuple[float, float, float, float, float, float]
NO_SHIFT: Shift = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def discount_factors(
    n_years: int, rate: np.ndarray, slope: float, llp: int, ltfr: float, convergence: float
) -> np.ndarray:
    """행마다 연 중간 시점 할인계수 (행 수 × n_years).

    최종관찰만기까지는 시장 금리(수준 + 기울기 적재)를 연 선도금리로 쓰고, 그 뒤 선도금리는
    LTFR로 지수 수렴한다. K-ICS Smith-Wilson 곡선을 단순화한 것이다 (W12). 그래서 금리 충격은
    최종관찰만기 뒤에서 약해지고, 최종관찰만기 연장의 효과는 시장 금리와 LTFR의 차이에 달려 있다.
    """
    k = np.arange(n_years)
    market = rate[:, None] + slope * slope_loading(k + 0.5)[None, :]
    if llp < n_years:
        last = market[:, llp - 1 : llp]
        decay = np.exp(-convergence * (k[llp:] - llp + 1))[None, :]
        market[:, llp:] = ltfr + (last - ltfr) * decay
    growth = np.cumprod(1.0 + market, axis=1)
    before = np.concatenate([np.ones((len(rate), 1)), growth[:, :-1]], axis=1)
    return 1.0 / (before * np.sqrt(1.0 + market))


def slope_loading(t: np.ndarray) -> np.ndarray:
    """기울기 1단위 변화가 만기 t의 할인율을 얼마나 바꾸는지. 3년 −1, 10년 0, 20년 이상 +10/7 (만기에 선형)."""
    return (np.minimum(t, 20.0) - 10.0) / 7.0


@dataclass(frozen=True)
class LiabilityBlock:
    """회사(또는 회사 묶음)의 장기보험 부채. 하위 블록마다 계리가정이 다르다. 금액 단위는 억원.

    evaluate()는 모든 모델포인트와 하위위험 시나리오 7개를 한 번에 배열로 계산한다.
    결과는 하위 블록마다 sub_risks()를 부른 것과 같다 (테스트로 확인).
    """

    blocks: Tuple[Tuple[Tuple[ModelPoint, ...], Assumptions], ...]

    @cached_property
    def _arrays(self) -> Dict[str, np.ndarray]:
        rows = [(mp, a, b) for b, (mps, a) in enumerate(self.blocks) for mp in mps]
        get = lambda f: np.array([f(mp, a) for mp, a, _ in rows], dtype=float)  # noqa: E731
        return {
            "block": np.array([b for _, _, b in rows]),
            "premium": get(lambda mp, a: mp.premium),
            "risk_share": get(lambda mp, a: mp.risk_share),
            "aging": get(lambda mp, a: mp.claim_aging),
            "death_share": get(lambda mp, a: mp.death_benefit_share),
            "sv": get(lambda mp, a: mp.surrender_value),
            "term": get(lambda mp, a: mp.term),
            "pay": get(lambda mp, a: mp.pay_years),
            "loss_ratio": get(lambda mp, a: a.loss_ratio),
            "lapse_pay": get(lambda mp, a: a.lapse_pay),
            "lapse_paid": get(lambda mp, a: a.lapse_paid),
            "expense_ratio": get(lambda mp, a: a.expense_ratio),
            "expense_paid": get(lambda mp, a: a.expense_paid),
            "inflation": get(lambda mp, a: a.inflation),
            "discount": get(lambda mp, a: a.discount),
            "death_rate": get(lambda mp, a: a.death_rate),
            "no_surrender": get(lambda mp, a: float(mp.no_surrender)),
        }

    @cached_property
    def first_year_claims(self) -> float:
        """기준 가정의 첫해 기대 보험금 합계 (억원). 재보험자산 크기의 대용에 쓴다."""
        return float(sum(cash_flows(mp, a)["claims"][0] for mps, a in self.blocks for mp in mps))

    @cached_property
    def pv_claims(self) -> float:
        """기준 가정의 장래 보험금 현가 (억원). 출재 시 재보험자산 크기의 대용."""
        total = 0.0
        for mps, a in self.blocks:
            for mp in mps:
                v = discount_factors(mp.term, np.array([a.discount]), 0.0, a.llp, a.ltfr, a.convergence)[0]
                total += float(cash_flows(mp, a)["claims"] @ v)
        return total

    @cached_property
    def first_year_risk_premium(self) -> float:
        """첫해 예정위험보험료 합계 (억원). 손해율(위험보험료 대비) 1단위 변화의 당해 보험금 변화다."""
        return float(sum(mp.premium * mp.risk_share for mps, a in self.blocks for mp in mps))

    @cached_property
    def first_year_lapse(self) -> float:
        """기준 가정의 첫해 해지율 (모델포인트별 |BEL| 가중 평균). 해지율 경험 변동의 당해 손실에 쓴다."""
        w = np.array([abs(bel(mp, a)) for mps, a in self.blocks for mp in mps])
        lapse = np.array([a.lapse_pay if mp.pay_years > 0 else a.lapse_paid for mps, a in self.blocks for mp in mps])
        return float(w @ lapse / w.sum())

    def best_estimate(self, shift: Shift = NO_SHIFT, llp: Optional[int] = None) -> float:
        """기준 시나리오 BEL 합계만 (억원). 하위위험이 필요 없는 신계약 수익성 비율에 쓴다 (계산 1/6)."""
        d_loss, lapse_mult, d_infl, d_rate, d_slope, nosv_mult = shift
        v = self._arrays
        m = lapse_mult * np.where(v["no_surrender"] > 0, nosv_mult, 1.0)
        a0 = self.blocks[0][1]
        curve = (a0.llp if llp is None else llp, a0.ltfr, a0.convergence)
        return float(
            _bel_batch(
                v["premium"], v["risk_share"], v["aging"], v["sv"], v["term"], v["pay"],
                v["loss_ratio"] + d_loss, v["lapse_pay"] * m, v["lapse_paid"] * m,
                v["expense_ratio"], v["expense_paid"], v["inflation"] + d_infl,
                v["discount"] + d_rate, v["death_rate"], d_slope, curve,
            ).sum()
        )

    def evaluate(self, shift: Shift = NO_SHIFT, llp: Optional[int] = None) -> Tuple[float, float]:
        """(보유계약 장래이익 = −BEL − 위험마진, 장기손해보험위험액). 세전, 억원."""
        d_loss, lapse_mult, d_infl, d_rate, d_slope, nosv_mult = shift
        v = self._arrays
        n = len(v["premium"])
        n_blocks = len(self.blocks)
        lr = v["loss_ratio"] + d_loss
        m = lapse_mult * np.where(v["no_surrender"] > 0, nosv_mult, 1.0)
        lp, lpd = v["lapse_pay"] * m, v["lapse_paid"] * m
        infl, disc = v["inflation"] + d_infl, v["discount"] + d_rate
        a0 = self.blocks[0][1]
        curve = (a0.llp if llp is None else llp, a0.ltfr, a0.convergence)
        er, ep, rs = v["expense_ratio"], v["expense_paid"], v["risk_share"]
        one = np.ones(n)
        # 시나리오: 기준, 사망, 장해·질병, 해지↑, 해지↓, 사업비
        scen = [
            (lr, rs, lp, lpd, er, ep, infl),
            (lr, rs * (1 + 0.125 * v["death_share"]), lp, lpd, er, ep, infl),
            (lr * 1.115, rs, lp, lpd, er, ep, infl),
            (lr, rs, lp * 1.35, lpd * 1.35, er, ep, infl),
            (lr, rs, lp * 0.65, lpd * 0.65, er, ep, infl),
            (lr, rs, lp, lpd, er * 1.1, ep * 1.1, infl + 0.01),
        ]
        cols = [np.concatenate([c[k] * one for c in scen]) for k in range(7)]
        tile = lambda arr: np.tile(arr, len(scen))  # noqa: E731
        bels = _bel_batch(
            tile(v["premium"]), cols[1], tile(v["aging"]), tile(v["sv"]), tile(v["term"]), tile(v["pay"]),
            cols[0], cols[2], cols[3], cols[4], cols[5], cols[6], tile(disc), tile(v["death_rate"]), d_slope,
            curve,
        ).reshape(len(scen), n)
        base, death_s, disease_s, up_s, down_s, exp_s = bels

        def per_block(x: np.ndarray) -> np.ndarray:
            return np.bincount(v["block"], weights=x, minlength=n_blocks)

        base_b = per_block(base)
        death = per_block(np.maximum(death_s - base, 0.0))
        disease = np.maximum(per_block(disease_s) - base_b, 0.0)
        up = np.maximum(per_block(up_s) - base_b, 0.0)
        down = np.maximum(per_block(down_s) - base_b, 0.0)
        mass = np.maximum(per_block(0.25 * (-base) + 0.25 * v["sv"] * v["premium"]), 0.0)
        expense = np.maximum(per_block(exp_s) - base_b, 0.0)
        subs = np.array([death.sum(), disease.sum(), np.maximum(np.maximum(up, down), mass).sum(), expense.sum()])
        long = float(np.sqrt(subs @ CORR_LONG4 @ subs))
        return -(float(base.sum()) + long * RM_FACTOR), long


def _bel_batch(
    premium, risk_share, aging, sv, term, pay, lr, lp, lpd, er, ep, infl, disc, death_rate, slope=0.0,
    curve=(23, 0.043, 0.1),
):
    """모델포인트·시나리오 행마다 BEL. cash_flows()와 같은 식을 배열로 쓴 것."""
    t = np.arange(int(term.max()))[None, :]
    alive = t < term[:, None]
    paying = t < pay[:, None]
    lapse = np.where(paying, lp[:, None], lpd[:, None])
    keep = np.cumprod(1.0 - lapse - death_rate[:, None], axis=1)
    survive = np.concatenate([np.ones((len(premium), 1)), keep[:, :-1]], axis=1)
    prem = np.where(paying, premium[:, None], 0.0) * survive
    claims = lr[:, None] * premium[:, None] * risk_share[:, None] * (1.0 + aging[:, None]) ** t * survive
    expense = np.where(paying, er[:, None], ep[:, None]) * premium[:, None] * (1.0 + infl[:, None]) ** t * survive
    surrender = lapse * survive * sv[:, None] * premium[:, None]
    disc_f = discount_factors(int(term.max()), disc, slope, *curve)
    return ((claims + expense + surrender - prem) * alive * disc_f).sum(axis=1)
