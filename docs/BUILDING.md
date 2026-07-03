# Building the documentation

Sphinx site for wbc-mjlab (layout follows [mjlab](https://github.com/mujocolab/mjlab)).

**Published:** https://wbc-mjlab.github.io/wbc-mjlab/

## Build locally

```bash
make sync
uv sync --group docs
make docs
```

Output: `docs/_build/index.html`

Live reload: `make docs-watch`

## Structure

All user-facing pages are **standalone reStructuredText** under `docs/source/`:

```
docs/
  conf.py
  index.rst
  source/
    concepts/           # modularity, RSI, presets, robots (conceptual)
    architecture.rst    # config pipeline
    tasks/              # task catalog + guides (User Guide)
    mdp/                # MDP API reference (User Guide)
    extensions/        # extensions how-to (User Guide)
    installation.rst
    usage.rst
    data.rst
    research.rst
    contributing.rst
    roadmap.rst
    visualization.rst   # Viser overlays, reference align, RSI panels
    workflows/
      training.rst      # resume, multi-GPU, mjlab CLI passthrough
    _static/
```

Edit `.rst` files directly — there are no markdown include wrappers.

## Deploy

Pushing to `main` runs `.github/workflows/docs.yml` → GitHub Pages.

**One-time setup:** repo **Settings → Pages → Build and deployment → Source = GitHub Actions**.
Do not use “Deploy from branch” with the `/docs` folder — that publishes this file via Jekyll instead of the Sphinx build.
