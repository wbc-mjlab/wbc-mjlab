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

The site follows [Diátaxis](https://diataxis.fr/): every page has **one** job, and the
sidebar groups pages by that job. Put new content in the section that matches what the reader
is doing — and link to it from the others instead of repeating it.

| Section | Reader is… | Pages |
|---|---|---|
| **Get started** | new, following along | `installation`, `workflows/quickstart`, `workflows/demo` |
| **How-to guides** | doing a task | `recipes` (index), `studio`, `data`, `tasks/` (choose / add a task), `workflows/training`, `visualization`, `workflows/deploy`, `extensions/` (add a robot), `troubleshooting` |
| **Concepts** | understanding | `concepts/` (modularity, presets and tasks, robots, adaptive sampling), `architecture` |
| **Reference** | looking up a fact | `usage` (CLI), `reference/` (configuration and files, Studio HTTP API, data formats), `mdp/` (terms), `api/` (autodoc) |
| **Project** | contributing, citing | `contributing`, `roadmap`, `research` |

```
docs/
  conf.py               theme, extensions, copy-button prompts
  index.rst             landing page + the sidebar (one toctree per section)
  source/
    installation.rst · workflows/ · recipes.rst · studio.rst · data.rst · visualization.rst
    tasks/              catalog (index), one page per task, adding.rst
    extensions/         add a robot: model files (custom_robots) or a package
    concepts/ · architecture.rst
    usage.rst           command-line reference
    reference/          configuration, studio_api, data_formats
    mdp/                shared MDP terms
    api/                autodoc
    _static/            css/custom.css (wbc-demo palette), refs.bib, logo
```

Page conventions:

- Open with one or two sentences on what the page is for; how-to guides list prerequisites in
  a `tip` admonition titled *You need* / *Before you start*.
- Show Studio and command-line paths side by side with `.. tab-set::` and
  `:sync-group: ui` (`:sync: studio` / `:sync: cli`) so the reader's choice sticks.
- End with a `.. seealso::` block (next steps), not a free-text "Related" line.
- Keep published URLs stable: move content between pages, not pages between folders.
- `make docs` must build with **no warnings**.

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
