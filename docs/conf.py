from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

project = "vgtr-py"
author = "langxin11"
copyright = "2026, langxin11"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_title = "vgtr-py"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/langxin11/vgtr_py",
            "icon": "fa-brands fa-github",
        }
    ],
}

autodoc_member_order = "bysource"
myst_enable_extensions = ["dollarmath"]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
