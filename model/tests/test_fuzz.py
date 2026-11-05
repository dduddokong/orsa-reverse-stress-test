"""무작위 경로 점검: 예외·비유한값이 없고 손해율 악화가 비율을 낮추는지 본다 (B 차선 최적화가 넓게 탐색하므로)."""
import numpy as np
import pytest

from interfaces.contracts import FACTORS, N_FACTORS, Regime
from model.archetypes import build_all
from model.insurer import StylizedInsurer

SCALE = np.array([0.03, 0.015, 0.03, 0.6, 0.1, 0.8, 0.02])


@pytest.fixture(scope="module")
def models():
    return {k: StylizedInsurer(p) for k, p in build_all().items()}


@pytest.mark.parametrize("key", ["A", "B", "C", "D"])
def test_random_paths_stay_finite_and_loss_ratio_monotone(models, key):
    rng = np.random.default_rng(42)
    m = models[key]
    loss = FACTORS.index("loss_ratio")
    for _ in range(40):
        b = m.initial()
        for _ in range(3):
            x = rng.uniform(-1, 1, N_FACTORS) * SCALE
            regime = list(Regime)[rng.integers(4)]
            actions = m.feasible_actions(b, x)
            a = actions[rng.integers(len(actions))]
            nb = m.step(b, x, regime, a)
            assert np.isfinite(nb.available_capital) and nb.required_capital > 0
            worse = x.copy()
            worse[loss] += 0.02
            assert m.step(b, worse, regime, a).ratio <= nb.ratio + 1e-9
            b = nb
