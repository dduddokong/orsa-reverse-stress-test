"""보험사 A~D 보정 요약표를 만든다 (원고 03절·B 차선 참고용).

    python -m model.report > docs/모형_보정표.md

표의 수치는 모형 출력이다. 원고에 쓰려면 results/REGISTRY.md에 등록하고 가정을 주의사항으로 붙인다.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from interfaces.contracts import FACTORS, N_FACTORS, NO_ACTION, Action, Regime

from .archetypes import ASSUMPTIONS, build_all
from .insurer import StylizedInsurer

ZERO = np.zeros(N_FACTORS)
SHOCKS = [
    ("금리 +100bp", {"rate_level": 0.01}),
    ("금리 −100bp", {"rate_level": -0.01}),
    ("기울기 +50bp", {"rate_slope": 0.005}),
    ("스프레드 +100bp", {"credit_spread": 0.01}),
    ("주가 −30%", {"equity": float(np.log(0.7))}),
    ("장기 손해율 +5%p", {"loss_ratio": 0.05}),
    ("해지율 +20%", {"lapse": 0.2}),
    ("사업비 인플레 +1%p", {"expense_infl": 0.01}),
]
ACTIONS = [
    ("출재 20%", Action(reinsurance_cession=0.2)),
    ("주식 30% 축소", Action(derisk=0.3)),
    ("후순위채 5%", Action(sub_debt=0.05)),
    ("배당 중단", Action(dividend_cut=True)),
]


def _x(spec: Dict[str, float]) -> np.ndarray:
    x = np.zeros(N_FACTORS)
    for k, v in spec.items():
        x[FACTORS.index(k)] = v
    return x


def _pp(v: float) -> str:
    return f"{100 * v:+.1f}"


def table() -> str:
    models = {k: StylizedInsurer(p) for k, p in build_all().items()}
    names = list(models)
    head = "| 항목 | " + " | ".join(names) + " |\n|---|" + "---|" * len(names) + "\n"
    lines: List[str] = ["# 보험사 A~D 보정표 (모형 출력)\n"]
    lines.append("`python -m model.report`로 생성. 기준 시점 2026-06-30. 비율 변화는 1년 이월 뒤 무충격 대비 %p.\n")

    rows = ["## 기준 시점\n", head]
    for label, f in (
        ("K-ICS 비율", lambda m: f"{100 * m.initial().ratio:.1f}%"),
        ("기본자본비율", lambda m: f"{100 * m.initial().basic_ratio:.1f}%"),
        ("가용자본 (조원)", lambda m: f"{m.initial().available_capital:.1f}"),
        ("요구자본 (조원)", lambda m: f"{m.initial().required_capital:.1f}"),
    ):
        rows.append(f"| {label} | " + " | ".join(f(models[n]) for n in names) + " |\n")
    base_paths = {}
    for n in names:
        m, b, path = models[n], models[n].initial(), []
        for _ in range(3):
            b = m.step(b, ZERO, Regime.NONE, NO_ACTION)
            path.append(f"{100 * b.ratio:.0f}")
        base_paths[n] = "→".join(path)
    rows.append("| 기준 경로 1·2·3년 (%) | " + " | ".join(base_paths[n] for n in names) + " |\n")
    lines += rows

    lines.append("\n## 1년 충격 반응 (K-ICS 비율 %p)\n")
    lines.append(head)
    for label, spec in SHOCKS + [("손해율 개편 (체제)", None), ("해지율 개편 (체제)", None)]:
        cells = []
        for n in names:
            m, b0 = models[n], models[n].initial()
            calm = m.step(b0, ZERO, Regime.NONE, NO_ACTION).ratio
            if spec is None:
                regime = Regime.LOSS if "손해율" in label else Regime.LAPSE
                hit = m.step(b0, ZERO, regime, NO_ACTION).ratio
            else:
                hit = m.step(b0, _x(spec), Regime.NONE, NO_ACTION).ratio
            cells.append(_pp(hit - calm))
        lines.append(f"| {label} | " + " | ".join(cells) + " |\n")

    stress = _x({"loss_ratio": 0.05, "credit_spread": 0.02})
    lines.append("\n## 경영대응 효과 (K-ICS 비율 %p, 평시 / 스트레스: 손해율 +5%p·스프레드 +200bp·손해율 개편)\n")
    lines.append(head)
    for label, a in ACTIONS:
        cells = []
        for n in names:
            m, b0 = models[n], models[n].initial()
            calm = m.step(b0, ZERO, Regime.NONE, a).ratio - m.step(b0, ZERO, Regime.NONE, NO_ACTION).ratio
            # 실행 가능 목록의 대응은 배당 중단을 함께 한다. 배당 여부는 빼고 비교한다.
            with_cut = Action(a.reinsurance_cession, a.derisk, a.sub_debt, dividend_cut=True)
            feasible = with_cut in m.feasible_actions(b0, stress)
            hit = m.step(b0, stress, Regime.LOSS, a).ratio - m.step(b0, stress, Regime.LOSS, NO_ACTION).ratio
            cells.append(f"{_pp(calm)} / {_pp(hit) if feasible else '불가'}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |\n")

    lines.append("\n## 경영대응 비용 (조원, 평시 / 스트레스, `action_cost`)\n")
    lines.append(head)
    for label, a in ACTIONS:
        cells = []
        for n in names:
            m, b0 = models[n], models[n].initial()
            c0 = sum(m.action_cost(b0, ZERO, a).values())
            c1 = sum(m.action_cost(b0, stress, a).values())
            cells.append(f"{c0:.2f} / {c1:.2f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |\n")

    lines.append("\n## D의 기초 비율 민감도 (3년 뒤 K-ICS 비율 %, 기준 경로 / 표준 스트레스)\n")
    lines.append("| 기초 비율 | 기준 경로 | 표준 스트레스 | 기초 기본자본비율 |\n|---|---|---|---|\n")
    from .sensitivity import outcomes

    for target in (1.40, 1.50, 1.70):
        m = StylizedInsurer(build_all(target_ratio_d=target)["D"])
        out = outcomes(m)
        lines.append(
            f"| {target:.0%} | {100 * out['base_y3']:.0f} | {100 * out['stress_y3']:.0f} | "
            f"{100 * m.initial().basic_ratio:.0f} |\n"
        )

    lines.append("\n## 체제 점프 정의와 출처\n")
    for n in names:
        src = models[n].p.sources
        lines.append(f"- **{n}** ({src['members']}): {src['regime']}\n")
        if "capital" in src:
            lines.append(f"  - {src['capital']}\n")
    lines.append("\n## 주요 가정 (`model/archetypes.py` ASSUMPTIONS)\n\n")
    for k in (
        "loan_duration",
        "security_duration",
        "va_pass_through",
        "regime_loss_csm_hit",
        "regime_lapse_csm_hit",
        "reinsurer_margin_share",
        "fire_sale_per_spread",
        "new_business_nosv_share",
        "tax_cap_headroom",
    ):
        lines.append(f"- `{k}` = {ASSUMPTIONS[k]:.4g}\n")
    lines.append("\n## 공시 대용 항목 (회사별)\n\n")
    for n in names:
        gaps = models[n].p.sources["gaps"]
        lines.append(f"- **{n}**: {gaps}\n")
    return "".join(lines)


if __name__ == "__main__":
    print(table())
