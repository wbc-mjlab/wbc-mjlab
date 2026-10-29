"""Two CPU ranks validate the same gradient/statistic helpers used on CUDA ranks."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _rank_worker(rank, rendezvous):
  import mjlab  # noqa: F401

  from wbc_mjlab.rl.wm.distributed import average_gradients, broadcast
  from wbc_mjlab.rl.wm.model import RunningNorm

  dist.init_process_group(
    "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
  )
  try:
    norm = RunningNorm(2, 1e-4)
    # Unequal local counts and masks catch incorrectly averaging per-rank means.
    if rank == 0:
      x, weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([1.0, 1.0])
    else:
      x, weight = (
        torch.tensor([[100.0, 100.0], [5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([0.0, 1.0, 1.0]),
      )
    norm.update(x, weight)
    assert norm.count == 4
    assert torch.allclose(norm.mean, torch.tensor([4.0, 5.0]))
    assert torch.allclose(norm.var, torch.tensor([5.0, 5.0]))
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
      model.weight.fill_(rank + 1.0)
    broadcast(model)
    assert torch.equal(model.weight, torch.ones_like(model.weight))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.full((1, 2), float(rank + 1))).sum().backward()
    average_gradients(model)
    assert torch.allclose(model.weight.grad, torch.full_like(model.weight, 1.5))
    opt.step()
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.85))
    # A second normalization update must also agree with the concatenated dataset.
    norm.update(torch.tensor([[9.0, 10.0]]))
    expected = torch.tensor(
      [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [9.0, 10.0]]
    )
    assert torch.allclose(norm.mean, expected.mean(0))
    assert torch.allclose(norm.var, expected.var(0, unbiased=False))
    # The critic must issue four gradient collectives on every rank even when
    # clip-end masking leaves one rank with fewer than four training rows.
    from types import SimpleNamespace

    from wbc_mjlab.rl.wm.algorithm import WorldModelRL

    critic = torch.nn.Linear(2, 1)
    broadcast(critic)
    owner = SimpleNamespace(
      gamma=0.99,
      lam=0.95,
      critic=critic,
      critic_norm=RunningNorm(2, 1e-4),
      critic_epochs=1,
      critic_mini_batches=4,
      device="cpu",
      max_grad_norm=1.0,
      critic_optimizer=torch.optim.SGD(critic.parameters(), lr=0.01),
    )
    inputs = torch.randn(4, 2, 2)
    mask = torch.ones(3, 2, dtype=torch.bool)
    if rank == 0:
      mask[1:] = False  # two rows versus six: includes empty minibatches
    loss = WorldModelRL._update_critic(
      owner, inputs, torch.zeros(4, 2), torch.ones(3, 2), torch.zeros(3, 2), mask
    )
    assert torch.isfinite(torch.tensor(loss))
    weight = critic.weight.detach().clone()
    dist.broadcast(weight, src=0)
    assert torch.allclose(critic.weight, weight)
  finally:
    dist.destroy_process_group()


def test_distributed_normalization_and_optimizer_agree(tmp_path):
  mp.spawn(_rank_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
