"""회사별 장기보험 모델포인트를 공시 하위위험액과 장래이익에 맞춘다.

맞추는 값(억원): 사망·장해질병·해지·사업비 위험액(정기경영공시), 장래이익 ≈ CSM,
무·저해지 해지율을 절반으로 낮추면 BEL이 CSM의 10.2% 늘어난다는 조건
(2024-11 해지율 가이드라인의 관측 영향: DB 4Q24 −11.5%, 현대 4Q24 −8.9%. W21),
그리고 부채 실효 듀레이션(공시 10~12년, 금리 민감도 식별용, 27·28회차).
자유 모수 8개: 보험료 규모, 표준형 위험보험료 비중, 납입 중 해지율, 유지비율, 표준형 위험률 상승률,
사망보험금 비중, 무해지 위험보험료 비중, 무해지 위험률 상승률.
모델포인트: 표준형 납입 중(15년납/45년, 환급금 있음), 무해지 납입 중(20년납/60년, 환급금 0), 완납.
결과는 model/data/liability_fit.json에 저장해 build_all()이 다시 맞추지 않게 한다.

    python -m model.fit_liability
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .liability import Assumptions, ModelPoint, bel, outflow_duration, portfolio_bel, sub_risks

FIT_PATH = Path(__file__).parent / "data" / "liability_fit.json"
TARGET_KEYS = ("death", "disease", "lapse", "expense", "pv_profit")
# 모델포인트 구조 (가정). 완납 블록 보험료는 납입 블록 합계의 30%. 보유계약 중 무해지 비중 35%
# (신계약 비중 2023 47%, 2024 상반기 64%보다 보유계약은 낮다고 봄 [가정]).
PAYING_TERM, PAYING_YEARS, PAID_TERM, PAID_SHARE = 45, 15, 35, 0.3
NOSV_TERM, NOSV_YEARS, NOSV_SHARE = 60, 20, 0.35
STD_SURRENDER = 1.0  # 표준형 해지 1건당 환급금 (연 보험료 대비) [가정]
LAPSE_REFORM_MULT, LAPSE_REFORM_CSM_HIT = 0.5, 0.102
# 부채 듀레이션(년): DB 10.3(2024-12)·11.8(2025-06), 현대 11.6(2025-12)·10.0(2026-06), 업권 10.3~12.5 (3·4회차 조사).
# 공시가 있는 회사는 최근 두 값의 평균, 없으면 11.0을 쓴다. 듀레이션 갭 규제·ALM 공시의 듀레이션은 금리 민감도 기준이라고
# 보고 실효 듀레이션으로 맞춘다 [정의 미확인].
LIABILITY_DURATION = 11.0
DURATION_BY_FIRM = {"DB손해보험": 11.05, "현대해상": 10.8}
MARKET_DISCOUNT = 0.041  # 2026-06 국고 10년 (ECOS). archetypes.ASSUMPTIONS["market_discount"]와 같게 둔다
BOUNDS = (
    [1e2, 0.05, 0.005, 0.01, -0.02, 0.0, 0.05, 0.0],
    [1e7, 0.8, 0.25, 0.4, 0.08, 1.0, 1.5, 0.08],  # 위험률 상승률 상한 8%: 건강보험 연령별 위험률 상승 [가정]
)


def build(theta: np.ndarray) -> Tuple[List[ModelPoint], Assumptions]:
    premium, risk_share, lapse, expense, aging, death_share, rs_nosv, aging_nosv = theta
    a = replace(
        Assumptions(), lapse_pay=lapse, expense_ratio=expense, expense_paid=expense / 3, discount=MARKET_DISCOUNT
    )
    mps = [
        ModelPoint(
            "standard", PAYING_TERM, PAYING_YEARS, (1 - NOSV_SHARE) * premium, risk_share,
            claim_aging=aging, surrender_value=STD_SURRENDER, death_benefit_share=death_share,
        ),
        ModelPoint(
            "no_surrender", NOSV_TERM, NOSV_YEARS, NOSV_SHARE * premium, rs_nosv,
            claim_aging=aging_nosv, death_benefit_share=death_share, no_surrender=True,
        ),
        ModelPoint(
            "paid", PAID_TERM, 0, PAID_SHARE * premium, risk_share,
            claim_aging=aging, death_benefit_share=death_share,
        ),
    ]
    return mps, a


def residuals(theta: np.ndarray, target: Dict[str, float]) -> np.ndarray:
    mps, a = build(theta)
    s = sub_risks(mps, a)
    s["pv_profit"] = -portfolio_bel(mps, a)
    out = [np.log(max(s[k], 1e-6) / target[k]) for k in TARGET_KEYS]
    nosv = mps[1]
    m = LAPSE_REFORM_MULT
    reform = bel(nosv, replace(a, lapse_pay=a.lapse_pay * m, lapse_paid=a.lapse_paid * m))
    out.append(5.0 * ((reform - bel(nosv, a)) / target["pv_profit"] - LAPSE_REFORM_CSM_HIT))
    # 부채 듀레이션: 모수 8개 중 위험률 상승률이 하위위험·CSM만으로는 식별되지 않아(26회차 점검) 함께 맞춘다.
    # 공시 듀레이션의 범위(유출만인지, 보험료 유입 포함인지)가 미확인이라 가중치 0.5로 덜 강하게 맞춘다.
    out.append(0.5 * np.log(outflow_duration(mps, a) / target.get("duration", LIABILITY_DURATION)))
    return np.array(out)


def fit(target: Dict[str, float], fixed: Optional[Dict[int, float]] = None) -> Dict[str, object]:
    """target에 맞춘다. fixed는 고정할 모수 {위치: 값} (공시가 적은 회사에서 식별되지 않는 모수를 묶는다)."""
    fixed = fixed or {}
    free = [i for i in range(len(BOUNDS[0])) if i not in fixed]

    def full(z: np.ndarray) -> np.ndarray:
        theta = np.empty(len(BOUNDS[0]))
        theta[free] = z
        for i, v in fixed.items():
            theta[i] = v
        return theta

    lo = [BOUNDS[0][i] for i in free]
    hi = [BOUNDS[1][i] for i in free]
    best = None
    scale = target["disease"] * 2
    for p0 in (scale, 3 * scale):
        for rs0 in (0.15, 0.35):
            for ag0 in (0.0, 0.03):
                x0 = np.array([p0, rs0, 0.06, 0.08, ag0, 0.1, 0.4, 0.04])[free]
                r = least_squares(lambda z: residuals(full(z), target), x0, bounds=(lo, hi))
                if best is None or r.cost < best.cost:
                    best = r
    theta = full(best.x)
    mps, a = build(theta)
    fitted = sub_risks(mps, a)
    fitted["pv_profit"] = -portfolio_bel(mps, a)
    return {
        "theta": [float(v) for v in theta],
        "fixed": {str(k): v for k, v in fixed.items()},
        "target": target,
        "fitted": {k: float(v) for k, v in fitted.items()},
        "max_abs_log_error": float(np.max(np.abs(best.fun))),
    }


def company_csm(d):
    from .archetypes import company_csm as _csm

    return _csm(d)


def targets_from_disclosure(d: Dict[str, float]) -> Dict[str, float]:
    csm = company_csm(d)
    return {
        "death": d["생명장기 하위위험-사망"],
        "disease": d["생명장기 하위위험-장해·질병"],
        "lapse": d["생명장기 하위위험-해지"],
        "expense": d["생명장기 하위위험-사업비"],
        "pv_profit": csm,  # 장래이익 대용치. 위험조정은 회사마다 공시 범위가 달라 뺐다
    }


def with_duration(target: Dict[str, float], name: str) -> Dict[str, float]:
    return dict(target, duration=DURATION_BY_FIRM.get(name, LIABILITY_DURATION))


def main() -> None:
    """하위위험 공시 4사(대형 3사 + 하나)는 공시값에, 나머지는 자기 장기위험액·CSM에 맞춘다.

    하위위험이 공시되지 않은 회사(삼성, 메리츠, D 편입사)는 하위위험 4개를 대형 3사 평균 비중 ×
    자기 장기손해보험위험액으로 두고, CSM은 자기 공시값을 쓴다 (W38). 그래서 수익성 모양은 회사마다 다르다.
    """
    from .archetypes import C_MEMBERS, D_MEMBERS, DETAILED, build_firm, company_csm, load, proxy_profile

    data = load()
    out = {}
    detailed = DETAILED + ("하나손해보험",)
    for name in detailed:
        out[name] = fit(with_duration(targets_from_disclosure(data[name]), name))
        out[name]["basis"] = "공시 하위위험"
        print(name, round(out[name]["max_abs_log_error"], 3))
    shares = np.mean(
        [
            [targets_from_disclosure(data[n])[k] / data[n]["요구자본-생명장기손해보험위험"] for k in TARGET_KEYS[:4]]
            for n in DETAILED
        ],
        axis=0,
    )
    proxy = proxy_profile(data)
    # 하위위험 미공시 회사는 해지율과 위험률 상승률이 식별되지 않아 하한에 붙는다(실행가능성 리뷰: 삼성 해지율 0.005).
    # 대형 3사 적합값의 평균으로 고정한다. 위험률 상승률이 하한(−2%)에 붙은 회사는 평균에서 뺀다 (리뷰5-1).
    lapse_avg = float(np.mean([out[n]["theta"][2] for n in DETAILED]))
    agings = [
        out[n]["theta"][4] for n in DETAILED if BOUNDS[0][4] + 1e-6 < out[n]["theta"][4] < BOUNDS[1][4] - 1e-6
    ]
    aging_avg = float(np.mean(agings))
    for name in sorted(set(C_MEMBERS + D_MEMBERS) - set(detailed)):
        firm = build_firm(name, data[name], proxy)
        target = dict(zip(TARGET_KEYS[:4], shares * firm.risks[0]))
        target["pv_profit"] = company_csm(data[name]) or firm.csm
        if target["pv_profit"] <= 0:
            continue  # CSM이 없는 회사(AXA 등)는 DB 모양을 줄여 쓴다
        out[name] = fit(with_duration(target, name), fixed={2: lapse_avg, 4: aging_avg})
        out[name]["basis"] = "대형 3사 평균 하위위험 비중 × 자기 장기위험액, 자기 CSM; 해지율·위험률 상승률은 대형 3사 평균 고정"
        print(name, round(out[name]["max_abs_log_error"], 3))
    FIT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
