"""B 차선 개발용 가짜 보험사 모형.

계수는 부호와 대략의 크기만 그럴듯하게 맞춘 임의값이다. 실제 결과나 원고에 쓰지 않는다.
A 차선의 model/ 구현이 계약 테스트를 통과하면 그것으로 교체한다.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .contracts import (
    NO_ACTION,
    Action,
    BalanceSheet,
    Regime,
    check_factors,
)

# 위험요인 1단위 편차당 가용자본 변화 (기초 가용자본 대비 비율)
_AC_SENS = np.array([4.0, 1.0, -6.0, 0.25, -2.5, -0.15, -3.0])
# 위험요인 1단위 편차당 요구자본 변화 (기초 요구자본 대비 비율)
_RC_SENS = np.array([-0.5, 0.0, 1.0, 0.1, 0.8, 0.05, 0.3])
_RATE_CONVEXITY = 40.0  # 금리 변화의 비선형(2차) 손실

_REGIME_AC_HIT = {
    Regime.NONE: 0.0,
    Regime.LOSS: 0.08,
    Regime.LAPSE: 0.05,
    Regime.BOTH: 0.13,
}
_REGIME_RC_RISE = {
    Regime.NONE: 0.0,
    Regime.LOSS: 0.04,
    Regime.LAPSE: 0.02,
    Regime.BOTH: 0.06,
}


class MockKicsModel:
    def __init__(
        self,
        available_capital: float = 10.0,
        required_capital: float = 5.0,
        base_profit: float = 0.03,
        dividend_payout: float = 0.01,
    ) -> None:
        self.ac0 = available_capital
        self.rc0 = required_capital
        self.base_profit = base_profit
        self.dividend_payout = dividend_payout

    def initial(self) -> BalanceSheet:
        return BalanceSheet(self.ac0, self.rc0, 0)

    def step(
        self,
        prev: BalanceSheet,
        x: np.ndarray,
        regime: Regime,
        action: Action,
    ) -> BalanceSheet:
        x = check_factors(x)
        ac, rc = prev.available_capital, prev.required_capital

        d_ac = ac * (
            self.base_profit
            + _AC_SENS @ x
            - _RATE_CONVEXITY * x[0] ** 2
            - _REGIME_AC_HIT[regime]
        )
        d_rc = rc * (_RC_SENS @ x + _REGIME_RC_RISE[regime])

        # 경영대응: 출재는 요구자본을 줄이지만 손해율이 높을수록 비싸다
        cession_cost = action.reinsurance_cession * (0.02 + 0.1 * max(x[4], 0.0))
        d_ac += -ac * cession_cost + ac * action.sub_debt
        if not action.dividend_cut:
            d_ac -= ac * self.dividend_payout
        rc_scale = (1 - 0.3 * action.reinsurance_cession) * (1 - 0.2 * action.derisk)

        new_ac = ac + d_ac
        new_rc = max((rc + d_rc) * rc_scale, 1e-6)
        return BalanceSheet(new_ac, new_rc, prev.t + 1)

    def feasible_actions(self, prev: BalanceSheet, x: np.ndarray) -> List[Action]:
        x = check_factors(x)
        actions = [NO_ACTION]
        for cession in (0.0, 0.2):
            for derisk in (0.0, 0.3):
                for sub_debt in (0.0, 0.05):
                    # 스프레드가 급등하거나 이미 100% 미만이면 후순위채 발행 불가
                    if sub_debt > 0 and (x[2] > 0.015 or prev.ratio < 1.0):
                        continue
                    a = Action(cession, derisk, sub_debt, dividend_cut=True)
                    actions.append(a)
        return actions
