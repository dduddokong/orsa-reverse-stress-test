"""K-ICS 지급여력기준금액 합산.

대분류 위험액을 상관행렬로 합산하고 운영위험액을 더해 기본요구자본을 만든 뒤 법인세조정액을 뺀다.
상관계수 값은 이 모듈에 두지 않는다. 출처가 붙은 값을 model/archetypes.py에서 넘긴다.
"""
from __future__ import annotations

import numpy as np

# 대분류 위험 순서. 상관행렬의 행·열 순서와 같다.
MAJOR_RISKS = ("insurance_long", "insurance_general", "market", "credit")
N_MAJOR = len(MAJOR_RISKS)


def check_corr(corr: np.ndarray) -> np.ndarray:
    """상관행렬이 대칭이고 대각이 1이며 양의 준정부호인지 검사한다."""
    c = np.asarray(corr, dtype=float)
    if c.shape != (N_MAJOR, N_MAJOR):
        raise ValueError(f"corr must have shape ({N_MAJOR}, {N_MAJOR}), got {c.shape}")
    if not np.allclose(c, c.T):
        raise ValueError("corr must be symmetric")
    if not np.allclose(np.diag(c), 1.0):
        raise ValueError("corr must have unit diagonal")
    if np.linalg.eigvalsh(c).min() < -1e-10:
        raise ValueError("corr must be positive semidefinite")
    return c


def aggregate(risks: np.ndarray, corr: np.ndarray) -> float:
    """분산효과 반영 후 대분류 위험액 합계 sqrt(r' C r)."""
    r = np.asarray(risks, dtype=float)
    return float(np.sqrt(r @ corr @ r))


def required_capital(
    risks: np.ndarray, corr: np.ndarray, operational: float, tax_rate: float, tax_cap: float
) -> float:
    """총요구자본 = 기본요구자본 - min(기본요구자본 × 평균세율, 한도) (별표22 Ⅳ.1-2, 제7장)."""
    basic = aggregate(risks, corr) + operational
    return basic - min(basic * tax_rate, tax_cap)
