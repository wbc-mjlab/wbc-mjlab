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

exclude_patterns = [
  "_build",
  "_templates",
  "Thumbs.db",
  ".DS_Store",
  "README.md",
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


def setup(app):
  app.connect("html-page-context", _add_wbc_quick_link_buttons, priority=502)

html_context = {
  "github_user": "wbc-mjlab",
  "github_repo": "wbc-mjlab",
  "github_version": "main",
  "doc_path": "docs",
}
