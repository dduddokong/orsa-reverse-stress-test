"""역사적 시장 변화로 모형의 비율 반응을 검증한다 (재현 검증).

시장 변화 Δ(기간 시작 → 끝)를 위험요인 x로 바꿔 기준 시점 보험사에 한 번 적용하고,
같은 기간 충격이 없을 때와의 차이를 "거시 요인 기여"로 본다. 회사 IR의 K-ICS 변동요인 분해
(예: 삼성화재 "거시지표")와 비교한다. 기준 시점(2026-06) 대차대조표에 과거 변화를 적용하므로 근사다.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from interfaces.contracts import FACTORS, N_FACTORS, NO_ACTION, Regime

from .insurer import InsurerParams, StylizedInsurer


def factor_change(start: Dict[str, float], end: Dict[str, float], equity_series: str = "kospi") -> np.ndarray:
    """시장 수준(%, 지수) 두 시점에서 위험요인 편차 x를 만든다. 금리·스프레드는 % → 소수."""
    x = np.zeros(N_FACTORS)
    x[FACTORS.index("rate_level")] = (end["ktb10"] - start["ktb10"]) / 100
    x[FACTORS.index("rate_slope")] = ((end["ktb10"] - end["ktb3"]) - (start["ktb10"] - start["ktb3"])) / 100
    x[FACTORS.index("credit_spread")] = ((end["aa3"] - end["ktb3"]) - (start["aa3"] - start["ktb3"])) / 100
    x[FACTORS.index("equity")] = float(np.log(end[equity_series] / start[equity_series]))
    return x


def macro_contribution(params: InsurerParams, x: np.ndarray) -> float:
    """충격 x의 K-ICS 비율 기여 (%p). 같은 1년 이월에서 x를 넣은 것과 뺀 것의 차이."""
    m = StylizedInsurer(params)
    b0 = m.initial()
    shocked = m.step(b0, x, Regime.NONE, NO_ACTION).ratio
    calm = m.step(b0, np.zeros(N_FACTORS), Regime.NONE, NO_ACTION).ratio
    return 100.0 * (shocked - calm)
