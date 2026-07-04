"""Sphinx configuration for wbc-mjlab (mjlab-style layout)."""

from __future__ import annotations

import os
import sys

import sphinx_book_theme

sys.path.insert(0, os.path.abspath("../src"))

project = "WBC-MJLab"
copyright = "2026, WBC-MJLab contributors"
author = "WBC-MJLab contributors"

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.napoleon",
  "sphinx.ext.viewcode",
  "sphinx.ext.githubpages",
  "sphinx.ext.intersphinx",
  "sphinx_copybutton",
  "sphinx_design",
  "sphinxcontrib.bibtex",
]

source_suffix = ".rst"

bibtex_bibfiles = ["source/_static/refs.bib"]

suppress_warnings = [
  "ref.bibtex",
]

intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
  "mjlab": ("https://mujocolab.github.io/mjlab/main/", None),
}

# Autodoc: live signatures/docstrings from source (mink/mjlab style).
autodoc_typehints = "signature"
autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
autodoc_default_options = {
  "members": True,
  "member-order": "bysource",
  "undoc-members": False,
  "show-inheritance": True,
  "exclude-members": "__init__, __post_init__, __new__",
}
# Keep the in-page TOC on section headings, not every method (mink).
toc_object_entries = False

# Mock the sim stack so autodoc does not need CUDA / full mjlab runtime.
# Mock mjlab itself (not only mujoco): a real mjlab + mocked mujoco breaks on
# type unions like ``int | mujoco.mjtJoint``.
autodoc_mock_imports = [
  "mjlab",
  "torch",
  "mujoco",
  "mujoco_warp",
  "viser",
  "mjviser",
  "wandb",
  "rsl_rl",
  "tensordict",
  "gymnasium",
  "warp",
  "prettytable",
  "tqdm",
  "trimesh",
  "hid",
  "h5py",
  "mediapy",
]

exclude_patterns = [
  "_build",
  "_templates",
  "Thumbs.db",
  ".DS_Store",
  "README.md",
  "BUILDING.md",
]

language = "en"

# Project site: https://wbc-mjlab.github.io/wbc-mjlab/
html_baseurl = "https://wbc-mjlab.github.io/wbc-mjlab/"

html_title = "WBC-MJLab Documentation"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
html_show_sphinx = False
html_last_updated_fmt = ""

html_static_path = ["source/_static"]
html_css_files = ["css/custom.css"]
html_favicon = "source/_static/wbc_mjlab_logo.png"

html_theme_options = {
  "path_to_docs": "docs/",
  "collapse_navigation": True,
  "repository_url": "https://github.com/wbc-mjlab/wbc-mjlab",
  "use_repository_button": True,
  "use_issues_button": True,
  "use_edit_page_button": True,
  "show_toc_level": 2,
  "logo": {
    "image_light": "wbc_mjlab_logo.png",
    "image_dark": "wbc_mjlab_logo.png",
    "text": "WBC-MJLab",
    "alt_text": "WBC-MJLab logo",
  },
}

WBC_QUICK_LINK_BUTTONS = [
  {
    "type": "link",
    "url": "https://wbc-mjlab.github.io/wbc-demo/",
    "tooltip": "Live demo",
    "icon": "fas fa-play",
    "text": "",
    "classes": "pst-navbar-icon wbc-quick-link",
  },
  {
    "type": "link",
    "url": "https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb",
    "tooltip": "Google Colab",
    "icon": "fab fa-google",
    "text": "",
    "classes": "pst-navbar-icon wbc-quick-link",
  },
  {
    "type": "link",
    "url": "https://pypi.org/project/wbc-mjlab/",
    "tooltip": "PyPI",
    "icon": "fas fa-box",
    "text": "",
    "classes": "pst-navbar-icon wbc-quick-link",
  },
]


def _add_wbc_quick_link_buttons(app, pagename, templatename, context, doctree):
  """Insert demo / Colab / PyPI next to the GitHub and download header buttons."""
  buttons = context.get("header_buttons", [])
  insert_at = 1
  if buttons and buttons[0].get("label") == "launch-buttons":
    insert_at = 2
  for offset, btn in enumerate(WBC_QUICK_LINK_BUTTONS):
    buttons.insert(insert_at + offset, btn)


def _skip_member(app, what, name, obj, skip, options):
  """Hide dataclass boilerplate (mjlab-style)."""
  if name in ("from_dict", "to_dict", "replace", "copy", "validate", "__post_init__"):
    return True
  return None


def _process_signature(app, what, name, obj, options, signature, return_annotation):
  """Suppress noisy __init__ signatures on config dataclasses."""
  if what == "class" and "exclude-members" in options:
    if "__init__" in options["exclude-members"]:
      return ("", None)
  return None


def _process_docstring(app, what, name, obj, options, lines):
  """Strip auto-generated dataclass docstrings (e.g. ``ClassName(*, ...)``)."""
  import dataclasses

  if what == "class" and dataclasses.is_dataclass(obj):
    if lines and lines[0].startswith(f"{obj.__name__}("):
      lines.clear()


def setup(app):
  app.connect("html-page-context", _add_wbc_quick_link_buttons, priority=502)
  app.connect("autodoc-skip-member", _skip_member)
  app.connect("autodoc-process-signature", _process_signature)
  app.connect("autodoc-process-docstring", _process_docstring)

html_context = {
  "github_user": "wbc-mjlab",
  "github_repo": "wbc-mjlab",
  "github_version": "main",
  "doc_path": "docs",
}
