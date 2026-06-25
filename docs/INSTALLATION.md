# Installation

**System requirements**

- **Training:** Linux + NVIDIA GPU (CUDA 12.8+ recommended, same as [mjlab](https://github.com/mujocolab/mjlab))
- **Evaluation:** Linux, macOS, or Windows (WSL) with `make sync-cpu`
- **Python:** 3.10+

[wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab) is an extension of
[mjlab](https://github.com/mujocolab/mjlab). Install mjlab's stack first via the
options below.

---

## Method 1 — Develop from source (uv, recommended)

For hacking on `wbc_mjlab` or running the bundled samples.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and sync

```bash
git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
make sync          # uv sync --extra cu128 --group dev
```

CPU-only / macOS evaluation:

```bash
make sync-cpu      # uv sync --extra cpu --group dev
```

### 3. Verify

```bash
uv run wbc-mjlab-list-envs
```

`uv run` uses the locked environment in `uv.lock` (same workflow as
`uv run demo` in mjlab).

---

## Method 2 — Use as a dependency in your own uv project

Add `wbc-mjlab` to an existing [uv](https://docs.astral.sh/uv/) project:

```bash
uv add wbc-mjlab mjlab
```

Or from GitHub:

```bash
uv add "wbc-mjlab @ git+https://github.com/wbc-mjlab/wbc-mjlab"
```

Editable local checkout:

```bash
uv add --editable /path/to/wbc_mjlab
```

Ensure your project also selects a CUDA or CPU extra for PyTorch (see mjlab
[installation guide](https://mujocolab.github.io/mjlab/main/source/installation.html)).

---

## Method 3 — Classic pip

```bash
pip install mjlab wbc-mjlab
```

Or editable from a clone:

```bash
git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
pip install -e .
```

You are responsible for installing a CUDA-capable PyTorch build when training on
GPU.

---

## Local mjlab checkout (optional)

When developing `wbc_mjlab` alongside a sibling `mjlab` repo, pin mjlab to your
local tree in `pyproject.toml` and re-lock:

```toml
[tool.uv.sources]
mjlab = { path = "../../mjlab", editable = true }
```

```bash
uv lock && make sync
```

Remove the override before publishing the lockfile for users who only install from
PyPI.

---

## After install — quickstart

```bash
uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples
uv run wbc-mjlab-train --task Wbc-G1 --dataset samples
uv run wbc-mjlab-play --task Wbc-G1 --dataset samples
```

See [data/g1/samples/README.md](../data/g1/samples/README.md) for bundled motion
credits.
