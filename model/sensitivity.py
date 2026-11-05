"""가정 민감도: archetypes.ASSUMPTIONS의 값을 하나씩 바꿔 대표 결과가 얼마나 움직이는지 본다 (W27).

대표 결과
- base_y3: 충격 없는 3년 뒤 K-ICS 비율
- stress_y3: 표준 스트레스 3년 뒤 비율 (1년차 손해율 개편 + 매년 손해율 +1%p, 스프레드 +100bp)
- cession_gap: 출재 20%의 비율 개선폭이 평시보다 스트레스에서 얼마나 작은지 (%p, H2 방향)

    python -m model.sensitivity
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from interfaces.contracts import FACTORS, N_FACTORS, NO_ACTION, Action, Regime

from . import archetypes
from .insurer import StylizedInsurer

ZERO = np.zeros(N_FACTORS)
STRESS = np.zeros(N_FACTORS)
STRESS[FACTORS.index("loss_ratio")] = 0.01
STRESS[FACTORS.index("credit_spread")] = 0.01
KEYS = (
    "loan_duration",
    "security_duration",
    "va_pass_through",
    "asset_slope_loading",
    "experience_credibility",
    "reinsurer_margin_share",
    "fire_sale_per_spread",
    "regime_loss_csm_hit",
    "regime_lapse_csm_hit",
    "payout_ratio_if_unknown",
    "tax_cap_headroom",
)


def outcomes(m: StylizedInsurer) -> Dict[str, float]:
    b = s = m.initial()
    for year in range(3):
        b = m.step(b, ZERO, Regime.NONE, NO_ACTION)
        s = m.step(s, STRESS, Regime.LOSS if year == 0 else Regime.NONE, NO_ACTION)
    b0 = m.initial()
    cede = Action(reinsurance_cession=0.2)

    def gain(x: np.ndarray, regime: Regime) -> float:
        return m.step(b0, x, regime, cede).ratio - m.step(b0, x, regime, NO_ACTION).ratio

    return {
        "base_y3": b.ratio,
        "stress_y3": s.ratio,
        "cession_gap": 100 * (gain(ZERO, Regime.NONE) - gain(STRESS, Regime.LOSS)),
    }


def run(keys: Sequence[str] = KEYS, factors: Sequence[float] = (0.5, 1.5)) -> pd.DataFrame:
    original = copy.deepcopy(archetypes.ASSUMPTIONS)
    rows: List[dict] = []
    try:
        base = {k: outcomes(StylizedInsurer(p)) for k, p in archetypes.build_all().items()}
        for key in keys:
            for f in factors:
                archetypes.ASSUMPTIONS.clear()
                archetypes.ASSUMPTIONS.update(copy.deepcopy(original))
                archetypes.ASSUMPTIONS[key] = original[key] * f
                for name, p in archetypes.build_all().items():
                    out = outcomes(StylizedInsurer(p))
                    for metric, v in out.items():
                        rows.append(
                            {
                                "assumption": key,
                                "factor": f,
                                "insurer": name,
                                "metric": metric,
                                "value": v,
                                "change": v - base[name][metric],
                            }
                        )
    finally:
        archetypes.ASSUMPTIONS.clear()
        archetypes.ASSUMPTIONS.update(original)
    return pd.DataFrame(rows)


def main() -> None:
    """민감도 표를 마크다운으로 출력한다: python -m model.sensitivity > docs/모형_민감도.md"""
    df = run()
    print("# 가정 민감도 (모형 출력)\n")
    print(
        "`python -m model.sensitivity`로 생성. 가정 하나를 ×0.5 / ×1.5로 바꿨을 때의 변화(%p). "
        "변화가 없는 가정은 표에서 뺐다.\n"
    )
    print("표준 스트레스: 1년차 손해율 개편 + 매년 손해율 +1%p·스프레드 +100bp, 3년.\n")
    for metric, title in (
        ("stress_y3", "표준 스트레스 3년 뒤 K-ICS 비율"),
        ("base_y3", "기준 경로 3년 뒤 K-ICS 비율"),
        ("cession_gap", "출재 20% 효과의 평시−스트레스 차이 (H2 방향)"),
    ):
        table = df[df.metric == metric].pivot_table(index=["assumption", "factor"], columns="insurer", values="change")
        scale = 1 if metric == "cession_gap" else 100
        table = (table * scale).round(1)
        table = table[(table.abs() > 0.05).any(axis=1)]
        print(f"## {title}\n")
        cols = list(table.columns)
        print("| 가정 | 배수 | " + " | ".join(cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for (key, f), row in table.iterrows():
            print(f"| `{key}` | ×{f:g} | " + " | ".join(f"{v:+.1f}" for v in row) + " |")
        print()


if __name__ == "__main__":
    main()
