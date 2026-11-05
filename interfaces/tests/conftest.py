import pytest

from interfaces.mock import MockKicsModel
from model.archetypes import build_all
from model.insurer import StylizedInsurer

_ARCHETYPES = build_all()

# A 차선 구현이 생기면 여기에 추가한다. 예: ("dbifc_model", lambda: SomeModel())
IMPLEMENTATIONS = [
    ("mock", MockKicsModel),
] + [(f"insurer_{k}", (lambda p=p: StylizedInsurer(p))) for k, p in _ARCHETYPES.items()]


@pytest.fixture(params=IMPLEMENTATIONS, ids=[name for name, _ in IMPLEMENTATIONS])
def model(request):
    _, factory = request.param
    return factory()
