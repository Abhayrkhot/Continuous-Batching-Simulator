from __future__ import annotations

import pytest

from batchsim.cost import LinearStepModel
from batchsim.request import RequestSpec
from batchsim.workload import Workload


@pytest.fixture
def unit_cost() -> LinearStepModel:
    """Every iteration takes exactly 1 s, so step counts read off as times."""
    return LinearStepModel.simple(const=1.0, per_token=0.0)


@pytest.fixture
def token_cost() -> LinearStepModel:
    """10 ms per iteration plus 0.1 ms per token."""
    return LinearStepModel.simple(const=0.01, per_token=1e-4)


def wl(*reqs: tuple[float, int, int]) -> Workload:
    """Workload from (arrival, prompt_len, output_len) triples."""
    return Workload(tuple(RequestSpec(a, p, o) for a, p, o in sorted(reqs)), "test")
