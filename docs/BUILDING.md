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

Live reload (rebuilds on edits under `docs/` **and** `src/`):

```bash
make docs-watch
```

Opens a local server (usually http://127.0.0.1:8000) and refreshes the browser when
you change RST or Python docstrings. Stop with Ctrl+C.

## Structure

User-facing pages are **standalone reStructuredText** under `docs/source/`.
Sidebar order: Getting Started → **Concepts** → User Guide → API Reference → Development.

```
docs/
  conf.py
  index.rst
  source/
    concepts/           # modularity, RSI, presets, robots (conceptual)
    architecture.rst    # config pipeline
    tasks/              # task catalog + guides (User Guide)
    mdp/                # Shared MDP design (User Guide)
    api/                # Live autodoc (extension, registry, presets, mdp, export)
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

## API reference (autodoc)

Live signatures come from Google-style docstrings via Sphinx autodoc:

- `docs/source/api/` — curated API index (`extension`, `mdp`)
- Narrative MDP pages also embed `autofunction` / `autoclass` for key symbols
- Enable/config: `sphinx.ext.autodoc`, `napoleon`, `viewcode` in `docs/conf.py`

When adding a public symbol, document it in source and list it on the matching
API page (or narrative page). Prefer explicit `autofunction` / `autoclass` over
blanket `automodule` so private helpers stay out.

## Deploy

Pushing to `main` runs `.github/workflows/docs.yml` → GitHub Pages.

**One-time setup:** repo **Settings → Pages → Build and deployment → Source = GitHub Actions**.
Do not use “Deploy from branch” with the `/docs` folder — that publishes this file via Jekyll instead of the Sphinx build.
