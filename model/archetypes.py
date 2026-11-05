"""공시로 복원한 가상 손보사 A~D (D-008, D-010, D-011).

입력은 model/data/disclosures_2026h1.csv (2026-06-30 기준, 별도, 출처 URL 포함).
공시가 없는 항목은 아래 ASSUMPTIONS의 가정으로 메우고, 각 모수의 근거를 InsurerParams.sources에 남긴다.

절차
1. 회사별로 위험액·자본·감응도를 절대금액(억원, 위험요인 1단위당)으로 만든다.
2. A·B는 한 회사, C·D는 여러 회사를 더한다(통합). 더한 뒤 AC 대비 상대값으로 바꾼다.
3. D는 기초 K-ICS 비율이 D_TARGET_RATIO가 되도록 자본을 비례 조정한다(D-011). 기본·보완자본 비중은 편입사 합계를 유지한다(D-016).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import json

import numpy as np
import pandas as pd

from interfaces.contracts import FACTORS, N_FACTORS, Regime

from .fit_liability import FIT_PATH
from .fit_liability import build as build_blocks
from .insurer import ActionTerms, InsurerParams
from .liability import LiabilityBlock
from .kics import N_MAJOR

DATA = Path(__file__).parent / "data" / "disclosures_2026h1.csv"
BASE_YEAR = 2026

# 별표22 <표3> 대분류 상관 (생명·장기, 일반손해, 시장, 신용). docs/KICS_표준모형_정리.md 2절.
CORR_MAJOR = np.array(
    [
        [1.00, 0.00, 0.25, 0.25],
        [0.00, 1.00, 0.25, 0.25],
        [0.25, 0.25, 1.00, 0.25],
        [0.25, 0.25, 0.25, 1.00],
    ]
)
# 별표22 <표6> 생명·장기 하위위험 중 공시되는 4개 (사망, 장해·질병, 해지, 사업비)
CORR_LONG = np.array(
    [
        [1.00, 0.25, 0.00, 0.25],
        [0.25, 1.00, 0.00, 0.50],
        [0.00, 0.00, 1.00, 0.50],
        [0.25, 0.50, 0.50, 1.00],
    ]
)
# 별표22 <표19> 시장 하위위험 (금리, 주식, 부동산, 외환)
CORR_MARKET = np.array(
    [
        [1.00, 0.25, 0.25, 0.25],
        [0.25, 1.00, 0.25, -0.25],
        [0.25, 0.25, 1.00, 0.25],
        [0.25, -0.25, 0.25, 1.00],
    ]
)

# 공시로 정할 수 없는 값. 바꾸면 모형개발일지에 회차와 이유를 적는다.
ASSUMPTIONS = {
    # 신용스프레드 노출: FISIS 특수채+금융채+회사채+외화표시유가증권+기업대출(2026-06, 회사별 34~57%).
    # 미확인 회사만 아래 비중을 쓴다. 외화표시유가증권에 주식이 섞여 있어 상한값이다.
    "spread_exposure_share": 0.5,
    # 스프레드 듀레이션: 대출 3.75년, 유가증권 6년(보험연구원 2018: 대출채권 3~4.5년, 유가증권 6년 이상)을
    # 회사별 FISIS 구성비로 가중한다. 구성 미확인이면 아래 평균값.
    # 검증(31회차, FY2025 사업보고서 만기분석 주석): DB 신용채권 수정듀레이션 5.2~8.6년(중앙 6.4),
    # 삼성 기업대출 3.3~5.1년(중앙 4.0). 5년 초과 구간의 대표만기 가정에 따라 범위가 넓다.
    "spread_duration": 6.0,
    "loan_duration": 3.75,
    "security_duration": 6.0,
    # 변동성조정(VA) 전가율: AA- 3년 스프레드 1bp 변화당 VA 변화(bp). 금감원 월별 할인율 산출 파일(2023-01~2026-09, 45개월)의
    # VA와 ECOS 스프레드로 회귀한 기울기는 월별 0.076(R² 0.19), 분기별 0.22이다. 2026년 자산군별 LP 분해로 본 기계적 상한은
    # 0.11~0.2다(회사채 비중 14~16%). VA = LP × 80%. 0.1로 두고 민감도는 0~0.2 (5회차 이전 가정 0.36은 과대였다).
    "va_pass_through": 0.1,
    # 자산 기울기 적재: 기울기(10년−3년) 1 변화가 자산 할인율을 얼마나 바꾸는지. 보험업권 채권 중 10년 초과물 64.5%
    # (뉴스)로 자산 키레이트가 10~30년에 몰려 있어, 부채와 같은 만기 선형 적재(20년 +1.43)의 절반 정도인 0.5로 둔다 [가정].
    "asset_slope_loading": 0.5,
    "assets_per_rc": None,  # 운용자산 미공시 회사: 대형 3사 평균 운용자산/요구자본 배수로 대체
    "equity_shock_avg": 0.42,  # 주식위험액 → 노출 환산 계수 (별표22 상장 35%·기타 49% 사이)
    "disease_shock_as_loss_ratio": 0.10,  # 장해·질병 충격(+13%/+10%)을 손해율 절대 변화로 환산
    "base_loss_ratio": 0.90,  # 장기 위험손해율 기준 수준 (삼성 IR 97.9% 참고, 업계 평균 [미확인])
    "lapse_shock": 0.35,  # 해지위험 옵션행사 충격 (별표22 2-6)
    "expense_inflation_share": 0.5,  # 사업비위험액 중 인플레이션 +1%p 몫
    # 기준 자본 생성 (W13): 규모 유지(D-009)에서 CSM은 신계약 ≈ 상각으로 거의 일정하므로
    # 연 ΔAC ≈ 당기순이익 − 배당 (Milliman 2022의 운영 자본 생성 정의를 단순화).
    # 순이익은 2025 연간과 2026 상반기 ×2의 평균 (FISIS 손익계산서 SDSI150V, 별도). 배당은 뉴스 수치다.
    "payout_ratio_if_unknown": 0.4,  # 배당 미확인 회사: 2025 배당/순이익 (삼성 0.41, DB 0.30, 메리츠 0.50) 평균
    # 법인세조정 한도 (W7): 별표22 Ⅳ 제7장 한도 = 순이연법인세부채 + 직전 5년 세전이익 × 50% × 평균세율.
    # 구성요소가 공시되지 않아 기준 시점 법인세조정액의 (1 + 여유)로 둔다. 큰 손실 경로에서 감세효과가 멈춘다.
    "tax_cap_headroom": 0.2,
    # 선형 0단계(use_cash_flow_liability=False)에서만 쓰는 체제 점프 크기 [가정]
    "regime_loss_equiv": 0.03,  # 손해율 개편 = 보유계약 손해율 +3%p
    "regime_lapse_risk_rise": 0.32,  # 해지율 개편 시 해지위험액 증가 (DB·KB·현대 25.4Q→26.2Q 평균 +31~33%)
    "regime_lapse_equiv": 0.10,  # 해지율 개편 = 해지율 10% 상대 상승
    # 할인율 최종관찰만기 일정: '26·'27년 23년, '28·'29년 24년, '35년 30년 (금융위 2025-10-17; 금감원 2026.10 안내).
    # 곡선 모형(liability.discount_factors)이 연장 효과를 직접 계산한다.
    "llp_schedule": {2026: 23, 2027: 23, 2028: 24, 2029: 24},
    "market_discount": 0.041,  # 2026-06 국고 10년 4.09% (ECOS). 변동성조정은 따로 넣지 않는다 [가정]
    # 1단계 현금흐름 부채를 쓸지. 쓰면 손해율·해지율·사업비 감응도와 체제 점프를 계리가정 변화로 다시 계산한다.
    "use_cash_flow_liability": True,
    # 신계약 중 무·저해지 비중: 2023 47%, 2024 상반기 63.8% (3회차 조사 G 메모). 최근값을 쓴다.
    "new_business_nosv_share": 0.638,
    # 손해율·해지율 경험 1년이 장래 가정에 반영되는 비율 (신뢰도 가중) [가정]. 체제 점프는 이와 별개로 전부 반영.
    "experience_credibility": 0.5,
    # 해지율 개편은 무·저해지 상품의 해지율을 낮추는 방향이다 (2024-11 가이드라인, W21).
    # 크기는 아래 regime_lapse_csm_hit에 맞춰 회사별로 푼다.
    # 체제 점프 크기 보정 (W17): 과거 가정 개편의 CSM 영향(세전, CSM 대비)에 맞춰 손해율 변화·해지율 배수를 푼다.
    # 손해율 개편: 삼성 4Q25 실손 −8.6%, 한화 1H26 −6.4%, DB 2Q26 −2.3% → 중앙값 6.4%
    # 해지율 개편(2024-11): DB 4Q24 −11.5%, 현대 4Q24 −8.9% → 평균 10.2%
    # 출처: 각사 IR·뉴스 (scratchpad round2, 2026-11-02). 일부는 계리가정 외 요인이 섞여 있다.
    "regime_loss_csm_hit": 0.064,
    "regime_lapse_csm_hit": 0.102,
    # 경영대응 (W9). 출재는 비례재보험(QS)으로 보아 출재율만큼 보험위험액이 준다(Asimit et al. 2015 구조).
    "reinsurer_margin_share": 0.3,  # 출재한 장래이익(세후 CSM) 중 재보험자 몫 [가정]
    "reinsurance_price_doubling_loss": 0.05,  # 손해율 +5%p면 출재 비용 2배 (Froot 2001 방향, 크기는 가정)
    "fire_sale_base": 0.005,  # 평시 매각 비용 (매각액 대비) [가정]
    "fire_sale_per_spread": 2.0,  # 스프레드 +100bp당 매각 할인 +2% (Ellul et al. 2011 방향, 크기는 가정)
    "fire_sale_per_equity_drop": 0.1,  # 주가 하락률 1당 매각 할인 +10% (−30%면 +3%) [가정]
    # 재보험자산 신용위험: 출재 보험금의 현가(현금흐름 부채에서 계산) × 일반기업 1~2등급 9~10년 계수 2.1% (별표22 <표30>)
    "reinsurer_credit_coefficient": 0.021,
    # 후순위채 쿠폰 5.6% − 운용수익률 3.16% (2024, 보험연구원 보고서 기사 경유, 원문 미확인)
    "sub_debt_negative_carry": 0.056 - 0.0316,
    # 팔 수 없는 전략적 주식 보유분 (기준 노출 대비). 0이면 제한 없음. 회사별 값은 사람이 정한다 [결정 필요]
    # A: 삼성전자 지분 약 22조원(2026-07-13 종가 기준 지분가치, 증권사 리포트) ÷ 주식 노출 약 37.3조원(IR 주식 비중
    # 33.6% × 운용자산) ≈ 0.59. 금산법·보험업법·지배구조 제약으로 스트레스 중 매각이 어렵다고 본다 [결정 필요 2].
    "min_equity_scale": {"A": 0.59, "B": 0.4, "C": 0.4, "D": 0.4},
    # 여러 해 대응 한도 (B 차선 시험에서 대응이 해마다 쌓여 내부 최대화가 늘 '전부'가 됨) [결정 필요]
    # 출재: 장기 건강보험 공동재보험 시장 용량을 감안해 누적 보유율 60% 하한. 후순위채: 3년 누적 기준 가용자본의 10%.
    # 주식(B·C·D): 기준 수량의 40%는 남긴다(A는 위 전략 지분 59%).
    "min_retention": 0.6,
    "max_sub_debt_total": 0.10,
    # 금리 하한: 시장 금리 0.5% 아래는 반영하지 않는다(기준 4.1%에서 누적 −3.6%p) [가정]
    "rate_floor": 0.005,
}
D_TARGET_RATIO = 1.50  # D-011 기준 시나리오. 민감도 1.40, 1.70
D_MEMBERS = ("한화손해보험", "롯데손해보험", "흥국화재", "NH농협손해보험", "하나손해보험", "AXA손해보험")
C_MEMBERS = ("삼성화재", "DB손해보험", "현대해상", "KB손해보험", "메리츠화재")
DETAILED = ("DB손해보험", "현대해상", "KB손해보험")  # 하위위험까지 공시된 대형사 (구성비 대용치의 기준)


def load(path: Path = DATA) -> Dict[str, Dict[str, float]]:
    """회사 → {항목: 값(억원 또는 %)}. 조원 단위는 억원으로 바꾼다."""
    df = pd.read_csv(path, encoding="utf-8")
    df["value"] = np.where(df["unit"] == "조원", df["value"] * 1e4, df["value"])
    out: Dict[str, Dict[str, float]] = {}
    for company, g in df.groupby("company"):
        out[company] = dict(zip(g["item"], g["value"]))
    return out


def _first(d: Dict[str, float], *keys: str) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if v is not None and np.isfinite(v):
            return float(v)
    return None


# 회사 공시 CSM을 FISIS 값보다 먼저 쓴다. FISIS는 보험계약자산으로 표시된 포트폴리오를 빼서 과소하다.
CSM_ITEMS = (
    "CSM 잔액(장기손해보험)",
    "CSM 잔액(장기원수 잔여보장)",
    "CSM 잔액(원수·수재 잔여보장)",
    "CSM 잔액(장기 보험부채 서술)",
    "CSM 잔액(기말)",
    "CSM 잔액(보험계약부채)",
)


def company_csm(d: Dict[str, float]) -> Optional[float]:
    return _first(d, *CSM_ITEMS)


def _grad_share(sub: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """sqrt(s'Cs)의 하위위험별 탄력성 (s_k ∂T/∂s_k / T). 합은 1."""
    total = float(np.sqrt(sub @ corr @ sub))
    return sub * (corr @ sub) / total**2


@dataclass
class Firm:
    """회사 하나의 절대금액 표현. 금액은 억원, 감응도는 위험요인 1단위당 억원."""

    name: str
    basic: float
    supp_uncapped: float  # 해약환급금 부족분 재분류 등 한도 밖 보완자본
    supp_capped: float  # 후순위채 등 50% 한도 안 보완자본
    risks: np.ndarray
    operational: float
    tax: float  # 법인세조정액 (양수)
    d_ac: np.ndarray  # dAC/dx
    d_ac_rate2: float  # d²AC/dr² / 2
    d_risk: np.ndarray  # d risk_i / dx, (N_MAJOR, N_FACTORS)
    long_subs: np.ndarray  # 사망, 장해·질병, 해지, 사업비
    new_csm: float  # 연환산 신계약 CSM
    tax_rate: float
    rc: float  # 공시 요구자본
    csm: float  # 공시 CSM 잔액
    net_income: float  # 연 당기순이익 (억원)
    dividend: float  # 연 배당 (억원)
    liability: tuple  # ((모델포인트들, 계리가정), ...) 억원
    equity_exposure: float
    notes: List[str] = field(default_factory=list)

    @property
    def ac(self) -> float:
        return self.basic + self.supp_uncapped + self.supp_capped


def _rate_fit(d: Dict[str, float]) -> Optional[np.ndarray]:
    """공시 금리 민감도(K-ICS %p)를 Δr에 대한 2차식 계수 (a, b)로 맞춘다."""
    pts = []
    for bp, keys in (
        (100, ("민감도: 금리 +100bp → K-ICS", "민감도: 국채금리 +100bp → K-ICS")),
        (50, ("민감도: 금리 +50bp → K-ICS",)),
        (-50, ("민감도: 금리 -50bp → K-ICS",)),
        (-100, ("민감도: 금리 -100bp → K-ICS", "민감도: 국채금리 -100bp → K-ICS")),
    ):
        v = _first(d, *keys)
        if v is not None:
            pts.append((bp / 1e4, v / 100.0))
    if len(pts) < 2:
        return None
    r = np.array([p[0] for p in pts])
    k = np.array([p[1] for p in pts])
    a, b = np.linalg.lstsq(np.column_stack([r, r**2]), k, rcond=None)[0]
    return np.array([a, b])


def _rate_grid(d: Dict[str, float]):
    """정기경영공시 금리 민감도 표(가용·요구자본, ±50·±100bp)를 (Δr, ΔAC, ΔRC) 배열로."""
    ac0 = _first(d, "가용자본(지급여력금액)", "가용자본(경과조치 전)")
    rc0 = _first(d, "요구자본(지급여력기준금액)", "요구자본(경과조치 전)")
    pts = []
    for bp in (-100, -50, 50, 100):
        ac = _first(d, f"금리 민감도 가용자본 @ {bp:+d}bp")
        rc = _first(d, f"금리 민감도 요구자본 @ {bp:+d}bp")
        if ac is None or rc is None:
            return None
        pts.append((bp / 1e4, ac - ac0, rc - rc0))
    arr = np.array(pts)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _drc_dmarket(risks: np.ndarray, op: float, tax_rate: float) -> float:
    """시장위험액 1억원 증가가 총요구자본을 얼마나 늘리는지 (분산효과·법인세 반영)."""
    agg = float(np.sqrt(risks @ CORR_MAJOR @ risks))
    return (1.0 - tax_rate) * float((CORR_MAJOR @ risks)[2]) / agg


def build_firm(name: str, d: Dict[str, float], proxy: Optional[dict] = None) -> Firm:
    """공시가 자세한 회사는 그대로, 아니면 proxy(대형 3사 평균 구성비)로 채운다."""
    A = ASSUMPTIONS
    notes: List[str] = []
    ac = _first(d, "가용자본(지급여력금액)", "가용자본", "가용자본(경과조치 전)")
    rc = _first(d, "요구자본(지급여력기준금액)", "요구자본(경과조치 전)")
    ratio0 = ac / rc

    majors = [
        _first(d, "요구자본-생명장기손해보험위험"),
        _first(d, "요구자본-일반손해보험위험"),
        _first(d, "요구자본-시장위험"),
        _first(d, "요구자본-신용위험"),
    ]
    op = _first(d, "요구자본-운영위험")
    if None in majors or op is None:
        # 구성 미공시: 대형 3사 평균 구성비로 기본요구자본을 나누고, 요구자본이 맞도록 크기를 정한다.
        shares, op_share, tax_rate = proxy["shares"], proxy["op_share"], proxy["tax_rate"]
        unit = float(np.sqrt(shares @ CORR_MAJOR @ shares)) + op_share
        basic_rc = rc / (1.0 - tax_rate)
        risks = shares * basic_rc / unit
        op = op_share * basic_rc / unit
        notes.append("위험액 구성 미공시: 대형 3사 평균 구성비 사용")
    else:
        risks = np.array(majors)
        basic_rc = float(np.sqrt(risks @ CORR_MAJOR @ risks)) + op
        tax_rate = (basic_rc - rc) / basic_rc  # 기타요구자본 포함
    tax = basic_rc * tax_rate

    # 자본 계층
    basic = _first(d, "기본자본")
    sub_debt = _first(d, "후순위채 잔액") or 0.0
    if basic is None:
        basic_share = proxy["basic_share"] if proxy else 0.5
        basic = ac * basic_share
        notes.append("기본자본 미공시: 대형 3사 평균 기본자본/가용자본 비율 사용")
    supp = ac - basic
    reclass = _first(d, "해약환급금 부족분 상당액 중 해약환급금 상당액 초과분(보완자본)")
    supp_capped = min(sub_debt, supp) if reclass is None else max(supp - reclass, 0.0)
    supp_uncapped = supp - supp_capped
    if reclass is None:
        notes.append("해약환급금 부족분 재분류 미공시: 보완자본 − 후순위채로 추정")

    # 하위위험 (장기: 사망, 장해·질병, 해지, 사업비 / 시장: 금리, 주식, 부동산, 외환)
    long_subs = np.array(
        [
            _first(d, "생명장기 하위위험-사망") or np.nan,
            _first(d, "생명장기 하위위험-장해·질병") or np.nan,
            _first(d, "생명장기 하위위험-해지") or np.nan,
            _first(d, "생명장기 하위위험-사업비") or np.nan,
        ]
    )
    if np.isnan(long_subs).any():
        long_subs = proxy["long_sub_shares"] * risks[0]
        notes.append("장기 하위위험 미공시: 대형 3사 평균 비중 사용")
    mkt_subs = np.array(
        [
            _first(d, "시장 하위위험-금리위험액") or np.nan,
            _first(d, "시장 하위위험-주식위험액") or np.nan,
            _first(d, "시장 하위위험-부동산위험액") or np.nan,
            _first(d, "시장 하위위험-외환위험액") or np.nan,
        ]
    )
    if np.isnan(mkt_subs).any():
        mkt_subs = proxy["mkt_sub_shares"] * risks[2]
        notes.append("시장 하위위험 미공시: 대형 3사 평균 비중 사용")

    assets = _first(d, "운용자산")
    if assets is None:
        assets = rc * proxy["assets_per_rc"]
        notes.append("운용자산 미공시: 대형사 평균 운용자산/요구자본 배수 사용")
    equity_share = _first(d, "운용자산 비중-주식")
    if equity_share is not None and name == "삼성화재":
        equity_exp = assets * equity_share / 100.0
    else:
        equity_exp = mkt_subs[1] / A["equity_shock_avg"]

    at = 1.0 - tax_rate
    d_ac = np.zeros(N_FACTORS)
    d_risk = np.zeros((N_MAJOR, N_FACTORS))
    rate = FACTORS.index("rate_level")
    grid = _rate_grid(d)
    if grid is not None:
        # 가용자본·요구자본 표가 있으면 나눠서 맞춘다 (W14). 요구자본 변화는 시장위험액 변화로 넣는다.
        r, d_ac_pts, d_rc_pts = grid
        a, b = np.linalg.lstsq(np.column_stack([r, r**2]), d_ac_pts, rcond=None)[0]
        d_ac[rate], d_ac_rate2 = a, b
        slope_rc = float(np.linalg.lstsq(r[:, None], d_rc_pts, rcond=None)[0][0])
        d_risk[2, rate] = slope_rc / _drc_dmarket(risks, op, tax_rate)
    else:
        fit = _rate_fit(d)
        if fit is None:
            fit = proxy["rate_fit_rel"] * ratio0
            notes.append("금리 민감도 미공시: 대형 3사 평균(비율 대비) 사용")
        # 비율 민감도만 있으면 전부 AC 변화로 본다 (표가 있는 4사는 요구자본 변화가 작다).
        d_ac[rate] = fit[0] * rc
        d_ac_rate2 = fit[1] * rc
        notes.append("금리 민감도: 비율 변화만 공시 → 전부 가용자본 변화로 처리")
    spread_assets = _first(d, "스프레드 민감 운용자산(core)")
    if spread_assets is None:
        spread_assets = assets * A["spread_exposure_share"]
        notes.append("스프레드 민감 자산 미확인: 운용자산 × 가정 비중")
    loans = _first(d, "신용자산-기업대출")
    secs = sum(_first(d, f"신용자산-{k}") or 0.0 for k in ("특수채", "금융채", "회사채", "외화표시유가증권"))
    if loans is not None and loans + secs > 0:
        duration = (loans * A["loan_duration"] + secs * A["security_duration"]) / (loans + secs)
    else:
        duration = A["spread_duration"]
        notes.append("신용자산 구성 미확인: 스프레드 듀레이션 평균값")
    d_ac[FACTORS.index("credit_spread")] = -spread_assets * duration * at
    d_ac[FACTORS.index("equity")] = equity_exp * at
    d_ac[FACTORS.index("loss_ratio")] = -long_subs[1] * at / A["disease_shock_as_loss_ratio"]
    d_ac[FACTORS.index("lapse")] = -long_subs[2] * at / A["lapse_shock"]
    d_ac[FACTORS.index("expense_infl")] = -long_subs[3] * at * A["expense_inflation_share"] / 0.01

    g_long = _grad_share(long_subs, CORR_LONG) * risks[0]  # 하위위험 1% 증가 → 장기위험액 증가(억원/100%)
    g_mkt = _grad_share(mkt_subs, CORR_MARKET) * risks[2]
    d_risk[0, FACTORS.index("loss_ratio")] = g_long[1] / A["base_loss_ratio"]
    d_risk[0, FACTORS.index("lapse")] = g_long[2]
    d_risk[2, FACTORS.index("equity")] = g_mkt[1]

    ni_obs = [v for v in (_first(d, "당기순이익(2025)"), 2.0 * (_first(d, "당기순이익(2026 1H)") or float("nan"))) if v is not None and np.isfinite(v)]
    if ni_obs:
        net_income = float(np.mean(ni_obs))
    else:
        net_income = proxy["ni_per_rc"] * rc
        notes.append("당기순이익 미확인: 대형사 평균 순이익/요구자본 사용")
    dividend = _first(d, "배당총액(2025)")
    if dividend is None:
        dividend = A["payout_ratio_if_unknown"] * max(net_income, 0.0)
        notes.append("배당 미확인: 배당성향 40% 가정")
    csm = company_csm(d)
    if csm is None:
        csm = proxy["csm_per_rc"] * rc
        notes.append("CSM 미공시: 대형사 평균 CSM/요구자본 사용")
    new_csm = 2.0 * (_first(d, "신계약 CSM(1H)", "신계약 CSM(1H 누적)") or 0.0)
    if new_csm == 0.0:
        # 신계약 CSM은 보유 CSM 규모에 비례한다고 본다 (W43: 요구자본 비례는 자본이 얇은 회사에서 과대)
        new_csm = proxy["new_csm_per_csm"] * csm
        notes.append("신계약 CSM 미공시: 대형사 평균 신계약 CSM/보유 CSM 사용")

    return Firm(
        name=name,
        basic=basic,
        supp_uncapped=supp_uncapped,
        supp_capped=supp_capped,
        risks=risks,
        operational=op,
        tax=tax,
        d_ac=d_ac,
        d_ac_rate2=d_ac_rate2,
        d_risk=d_risk,
        long_subs=long_subs,
        new_csm=new_csm,
        tax_rate=tax_rate,
        rc=rc,
        csm=csm,
        net_income=net_income,
        dividend=dividend,
        liability=_liability_blocks(name, risks[0], proxy),
        equity_exposure=equity_exp,
        notes=notes,
    )


def _liability_blocks(name: str, long_risk_major: float, proxy: Optional[dict]) -> tuple:
    """맞춘 모델포인트가 있으면 그대로, 없으면 DB 모양을 장기위험액 비율로 늘리거나 줄인다."""
    fits = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    if name in fits:
        theta = np.array(fits[name]["theta"])
    else:
        ref = fits["DB손해보험"]
        theta = np.array(ref["theta"])
        theta[0] *= long_risk_major / proxy["db_long_major"]
    mps, a = build_blocks(theta)
    return ((tuple(mps), a),)


def _new_business_points(mps: tuple, a) -> tuple:
    """보유계약의 표준형·무해지 납입 모델포인트를 신계약 구성비(무해지 비중)로 다시 섞는다."""
    from dataclasses import replace

    std, nosv = mps[0], mps[1]
    total = std.premium + nosv.premium
    w = ASSUMPTIONS["new_business_nosv_share"]
    return ((replace(std, premium=(1 - w) * total), replace(nosv, premium=w * total)), a)


def _solve_shift(block: LiabilityBlock, target_loss: float, make, lo: float, hi: float) -> float:
    """장래이익이 target_loss만큼 줄어드는 가정 변화 크기를 이분법으로 찾는다."""
    from scipy.optimize import brentq

    of0, _ = block.evaluate()

    def gap(v: float) -> float:
        return (of0 - block.evaluate(make(v))[0]) - target_loss

    if gap(hi) < 0:
        return hi
    return float(brentq(gap, lo, hi))


def proxy_profile(data: Dict[str, Dict[str, float]]) -> dict:
    """하위위험까지 공시된 대형 3사의 평균 구성비. 미공시 회사의 대용치로 쓴다."""
    firms = [build_firm(n, data[n], proxy=None) for n in DETAILED]
    basic_rcs = [np.sqrt(f.risks @ CORR_MAJOR @ f.risks) + f.operational for f in firms]

    def mean(vals):
        return np.mean(np.array(vals, dtype=float), axis=0)

    mkt_rows = []
    for n in DETAILED:
        d = data[n]
        m = np.array(
            [
                d["시장 하위위험-금리위험액"],
                d["시장 하위위험-주식위험액"],
                d["시장 하위위험-부동산위험액"],
                d["시장 하위위험-외환위험액"],
            ]
        )
        mkt_rows.append(m / d["요구자본-시장위험"])
    rcs = [data[n]["요구자본(지급여력기준금액)"] for n in DETAILED]
    return {
        "shares": mean([f.risks / b for f, b in zip(firms, basic_rcs)]),
        "op_share": mean([f.operational / b for f, b in zip(firms, basic_rcs)]),
        "tax_rate": mean([f.tax_rate for f in firms]),
        "basic_share": mean([f.basic / f.ac for f in firms]),
        "long_sub_shares": mean([f.long_subs / f.risks[0] for f in firms]),
        "mkt_sub_shares": mean(mkt_rows),
        "assets_per_rc": mean([data[n]["운용자산"] / rc for n, rc in zip(DETAILED, rcs)]),
        "rate_fit_rel": mean([_rate_fit(data[n]) / (f.ac / rc) for n, f, rc in zip(DETAILED, firms, rcs)]),
        "new_csm_per_csm": mean([f.new_csm / f.csm for f in firms]),
        "csm_per_rc": mean([f.csm / rc for f, rc in zip(firms, rcs)]),
        "ni_per_rc": mean([f.net_income / rc for f, rc in zip(firms, rcs)]),
        "db_long_major": data["DB손해보험"]["요구자본-생명장기손해보험위험"],
    }


def combine(name: str, firms: Sequence[Firm], target_ratio: Optional[float] = None) -> InsurerParams:
    """회사들을 더해 InsurerParams로 만든다. target_ratio가 있으면 기본자본으로 비율을 맞춘다.

    위험액을 더한 뒤 다시 합산하면 회사 간 분산효과가 생겨 요구자본이 각사 합보다 작아진다.
    통합은 가상 단일회사가 아니라 업권 합계를 뜻하므로, 그 차이를 기타요구자본으로 되돌린다.
    """
    A = ASSUMPTIONS
    risks = sum(f.risks for f in firms)
    op = sum(f.operational for f in firms)
    tax = sum(f.tax for f in firms)
    basic_rc = float(np.sqrt(risks @ CORR_MAJOR @ risks)) + op
    tax_rate = tax / basic_rc
    rc_sum = sum(f.rc for f in firms)
    other = rc_sum - (basic_rc - tax)
    rc = rc_sum
    basic = sum(f.basic for f in firms)
    supp_uncapped = sum(f.supp_uncapped for f in firms)
    supp_capped = sum(f.supp_capped for f in firms)
    supp_recognised = supp_uncapped + min(supp_capped, 0.5 * rc)
    if target_ratio is not None:
        # 자본 계층 비중은 편입사 합계를 유지하고 크기만 줄인다 (D-016: 기본자본비율은 출발 상태가 갖출 전제).
        scale = target_ratio * rc / (basic + supp_uncapped + supp_capped)
        if scale * supp_capped > 0.5 * rc:
            scale = (target_ratio - 0.5) * rc / (basic + supp_uncapped)
        basic, supp_uncapped, supp_capped = basic * scale, supp_uncapped * scale, supp_capped * scale
        supp_recognised = supp_uncapped + min(supp_capped, 0.5 * rc)
    ac = basic + supp_recognised

    d_ac = sum(f.d_ac for f in firms) / ac
    rate2 = sum(f.d_ac_rate2 for f in firms) / ac
    d_risk = sum(f.d_risk for f in firms) / risks[:, None]
    new_csm = sum(f.new_csm for f in firms)

    loss, lapse = FACTORS.index("loss_ratio"), FACTORS.index("lapse")
    regime_ac = {
        Regime.NONE: 0.0,
        Regime.LOSS: -d_ac[loss] * A["regime_loss_equiv"],
        Regime.LAPSE: -d_ac[lapse] * A["regime_lapse_equiv"],
    }
    regime_ac[Regime.BOTH] = regime_ac[Regime.LOSS] + regime_ac[Regime.LAPSE]
    lapse_rise = np.zeros(N_MAJOR)
    lapse_rise[0] = d_risk[0, lapse] * A["regime_lapse_risk_rise"]
    loss_rise = d_risk[:, loss] * A["regime_loss_equiv"]
    regime_risk = {
        Regime.NONE: np.zeros(N_MAJOR),
        Regime.LOSS: loss_rise,
        Regime.LAPSE: lapse_rise,
        Regime.BOTH: loss_rise + lapse_rise,
    }

    at = 1.0 - tax_rate
    cession_cost = A["reinsurer_margin_share"] * sum(f.csm for f in firms) * at / ac
    equity_exp = sum(f.equity_exposure for f in firms)
    actions = ActionTerms(
        cession_relief=1.0,
        cession_cost=cession_cost,
        cession_cost_per_loss=cession_cost / A["reinsurance_price_doubling_loss"],
        derisk_relief=float(d_risk[2, FACTORS.index("equity")]),
        derisk_cost=A["fire_sale_base"] * equity_exp / ac,
        derisk_cost_per_spread=A["fire_sale_per_spread"] * equity_exp / ac,
        derisk_cost_per_equity_drop=A["fire_sale_per_equity_drop"] * equity_exp / ac,
        cession_credit=(
            sum(LiabilityBlock(f.liability).pv_claims for f in firms)
            * A["reinsurer_credit_coefficient"] * 1e-4  # 억원 → 조원
        ),
        sub_debt_carry=A["sub_debt_negative_carry"] * at,
        min_equity_scale=A["min_equity_scale"].get(name, 0.0),
        min_retention=A["min_retention"],
        max_sub_debt_total=A["max_sub_debt_total"],
    )
    liability = None
    new_business = None
    regime_shift = {}
    regime_note = "선형 0단계: regime_loss_equiv, regime_lapse_equiv"
    if A["use_cash_flow_liability"]:
        liability = LiabilityBlock(tuple(b for f in firms for b in f.liability))
        # 신계약 = 각사 납입 중 모델포인트(표준형·무해지)를 신계약 구성비로 새로 판다고 본다 (수익성 비율만 쓴다, W22).
        new_business = LiabilityBlock(tuple(_new_business_points(mps, a) for f in firms for mps, a in f.liability))
        # 공시 금리 민감도는 자산·부채 순효과다. 부채 몫(세후)을 빼 자산 몫만 ac_sens에 남긴다 (W23).
        of0, _ = liability.evaluate()
        of_up, _ = liability.evaluate((0.0, 1.0, 0.0, 0.01, 0.0, 1.0))
        of_dn, _ = liability.evaluate((0.0, 1.0, 0.0, -0.01, 0.0, 1.0))
        rate = FACTORS.index("rate_level")
        d_ac[rate] -= (of_up - of_dn) / 0.02 * at / ac
        rate2 -= (of_up + of_dn - 2.0 * of0) / (2.0 * 1e-4) * at / ac
        # 자산 쪽 기울기 노출: 자산 금리 감응도 × 자산 키레이트 적재 (W30, 실행가능성 리뷰 ⑦)
        d_ac[FACTORS.index("rate_slope")] = d_ac[rate] * A["asset_slope_loading"]
        # 공시 요구자본 민감도에도 부채 할인 변화에 따른 장기위험액 변화가 들어 있다. 그 몫을 시장위험 감응도에서 뺀다.
        _, long_up = liability.evaluate((0.0, 1.0, 0.0, 0.01, 0.0, 1.0))
        _, long_dn = liability.evaluate((0.0, 1.0, 0.0, -0.01, 0.0, 1.0))
        agg = float(np.sqrt(risks @ CORR_MAJOR @ risks))
        grad = CORR_MAJOR @ risks / agg  # ∂(분산 후 합계)/∂위험액
        d_risk[2, rate] -= (long_up - long_dn) / 0.02 * grad[0] / grad[2] / risks[2]
        for col in (loss, lapse, FACTORS.index("expense_infl")):
            d_ac[col] = 0.0
        d_risk[0, loss] = d_risk[0, lapse] = 0.0
        regime_ac = {r: 0.0 for r in Regime}
        regime_risk = {r: np.zeros(N_MAJOR) for r in Regime}
        csm_total = sum(f.csm for f in firms)
        d_lr = _solve_shift(liability, csm_total * A["regime_loss_csm_hit"], lambda v: (v, 1.0, 0.0, 0.0, 0.0, 1.0), 0.0, 0.2)
        # 해지율 개편 = 무·저해지 해지율을 낮춤 (2024-11 가이드라인 방향). 감소폭 v를 0~0.95에서 푼다.
        cut = _solve_shift(
            liability, csm_total * A["regime_lapse_csm_hit"], lambda v: (0.0, 1.0, 0.0, 0.0, 0.0, 1.0 - v), 0.0, 0.95
        )
        m_lapse = 1.0 - cut
        regime_shift = {
            Regime.NONE: (0.0, 1.0, 1.0),
            Regime.LOSS: (d_lr, 1.0, 1.0),
            Regime.LAPSE: (0.0, 1.0, m_lapse),
            Regime.BOTH: (d_lr, 1.0, m_lapse),
        }
        regime_note = (
            f"손해율 개편 = 손해율 +{d_lr:.2%}, 해지율 개편 = 무·저해지 해지율 ×{m_lapse:.2f} "
            f"(CSM 대비 −{A['regime_loss_csm_hit']:.1%}, −{A['regime_lapse_csm_hit']:.1%}에 맞춤)"
        )
    notes = sorted({f"{f.name}: {n}" for f in firms for n in f.notes})
    sources = {
        "data": "model/data/disclosures_2026h1.csv (2026-06-30, 각사 정기경영공시·IR·FISIS)",
        "corr": "보험업감독업무시행세칙 [별표22] <표3>·<표6>·<표19>",
        "members": ", ".join(f.name for f in firms),
        "assumptions": "model/archetypes.py ASSUMPTIONS",
        "gaps": "; ".join(notes) or "없음",
    }
    sources["regime"] = regime_note
    if target_ratio is not None:
        sources["capital"] = f"편입사 자본 구성비를 유지한 채 크기를 조정해 기초 K-ICS 비율을 {target_ratio:.0%}로 설정 (D-011·D-016, 관측치 아님)"

    # 단위: 억원 → 조원
    k = 1e-4
    return InsurerParams(
        name=name,
        base_year=BASE_YEAR,
        basic_capital=basic * k,
        supplementary_capital=(supp_uncapped + supp_capped) * k,
        uncapped_supplementary=supp_uncapped * k,
        risks=risks * k,
        corr=CORR_MAJOR,
        operational=op * k,
        tax_rate=tax_rate,
        tax_cap=tax * (1.0 + A["tax_cap_headroom"]) * k,
        other_required=other * k,
        llp_schedule=A["llp_schedule"] if A["use_cash_flow_liability"] else {},
        min_rate_level=A["rate_floor"] - A["market_discount"],
        actions=actions,
        liability=liability,
        regime_shift=regime_shift,
        ac_growth=sum(f.net_income for f in firms) / ac,
        new_business_growth=new_csm * (1.0 - tax_rate) / ac,
        va_pass_through=A["va_pass_through"] if A["use_cash_flow_liability"] else 0.0,
        experience_credibility=A["experience_credibility"] if A["use_cash_flow_liability"] else 1.0,
        new_business=new_business,
        rc_growth=0.0,  # D-009: 규모 유지
        dividend_payout=sum(f.dividend for f in firms) / ac,
        ac_sens=d_ac,
        ac_rate_convexity=rate2,
        risk_sens=d_risk,
        regime_ac_hit=regime_ac,
        regime_risk_rise=regime_risk,
        sources=sources,
    )


def build_all(target_ratio_d: float = D_TARGET_RATIO, path: Path = DATA) -> Dict[str, InsurerParams]:
    data = load(path)
    proxy = proxy_profile(data)

    def firm(n: str) -> Firm:
        return build_firm(n, data[n], proxy)

    return {
        "A": combine("A", [firm("삼성화재")]),
        "B": combine("B", [firm("DB손해보험")]),
        "C": combine("C", [firm(n) for n in C_MEMBERS]),
        "D": combine("D", [firm(n) for n in D_MEMBERS], target_ratio=target_ratio_d),
    }
