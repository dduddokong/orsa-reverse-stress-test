"""A·B 차선이 공유하는 인터페이스 계약.

공동 소유 파일이다. DECISIONS.md의 결정과 양쪽 승인 없이 바꾸지 않는다.
계약을 바꾸면 interfaces/tests/도 같은 PR에서 바꾼다.

시간 단위는 1년(D-003). 한 기간의 위험요인 x는 기준 전망 대비 편차이며
단위는 FACTOR_UNITS를 따른다(D-004).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

FACTORS = (
    "rate_level",
    "rate_slope",
    "credit_spread",
    "equity",
    "loss_ratio",
    "lapse",
    "expense_infl",
)
N_FACTORS = len(FACTORS)

FACTOR_UNITS = {
    "rate_level": "국고채 10년 금리 변화, 소수 (0.01 = +100bp)",
    "rate_slope": "10년-3년 스프레드 변화, 소수",
    "credit_spread": "회사채(AA-) 신용스프레드 변화, 소수",
    "equity": "KOSPI 연간 로그수익률의 기준 대비 편차",
    "loss_ratio": "장기보험 손해율 변화, 소수 (0.05 = +5%p)",
    "lapse": "해지율의 상대 변화 (0.2 = 기준 해지율의 1.2배)",
    "expense_infl": "사업비 인플레이션 변화, 소수",
}

KICS_RECOMMENDED = 1.30  # 권고기준 130%
KICS_MINIMUM = 1.00  # 조치기준 100%


class Regime(Enum):
    """한 해 동안 일어나는 계리 체제 점프."""

    NONE = "none"
    LOSS = "loss_ratio_reform"
    LAPSE = "lapse_reform"
    BOTH = "both"


@dataclass(frozen=True)
class Action:
    """한 해 동안 실행하는 경영대응. 기본값은 대응 없음."""

    reinsurance_cession: float = 0.0  # 추가 출재율, 0~1
    derisk: float = 0.0  # 위험자산 축소 비중, 0~1
    sub_debt: float = 0.0  # 후순위채 발행액, 기초 가용자본 대비 비율
    dividend_cut: bool = False


NO_ACTION = Action()


@dataclass(frozen=True)
class BalanceSheet:
    """K-ICS 관점의 요약 대차대조표. 금액 단위는 조원."""

    available_capital: float  # 가용자본
    required_capital: float  # 요구자본 (분산효과 반영 후)
    t: int = 0  # 경과 연수

    @property
    def ratio(self) -> float:
        return self.available_capital / self.required_capital


@runtime_checkable
class KicsModel(Protocol):
    """A 차선이 구현하고 B 차선이 사용하는 보험사 모형."""

    def initial(self) -> BalanceSheet:
        """기준 시점(t=0) 대차대조표."""
        ...

    def step(
        self,
        prev: BalanceSheet,
        x: np.ndarray,
        regime: Regime,
        action: Action,
    ) -> BalanceSheet:
        """한 해 이월. x는 길이 N_FACTORS의 편차 벡터. 결정적이어야 한다."""
        ...

    def feasible_actions(self, prev: BalanceSheet, x: np.ndarray) -> Sequence[Action]:
        """시나리오 x에서 실행 가능한 대응 목록. NO_ACTION을 항상 포함한다."""
        ...


def check_factors(x: np.ndarray) -> np.ndarray:
    """x를 float 배열로 바꾸고 모양과 유한성을 검사한다."""
    arr = np.asarray(x, dtype=float)
    if arr.shape != (N_FACTORS,):
        raise ValueError(f"x must have shape ({N_FACTORS},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must be finite")
    return arr
