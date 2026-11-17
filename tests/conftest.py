import pytest


@pytest.fixture(autouse=True)
def _fresh_dynamo():
  """Every algorithm recompiles the same methods; without a reset, tests sharing one
  process reach dynamo's recompile limit and run them eagerly inside graph capture."""
  import torch

  torch._dynamo.reset()
  yield
