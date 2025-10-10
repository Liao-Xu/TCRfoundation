# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'TCRfoundation'
author = 'Xu Liao'
copyright = '2025, Xu Liao'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_nb',               # <-- use MyST-NB (do NOT also load myst_parser or nbsphinx)
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']


# -- Autodoc / Napoleon ------------------------------------------------------
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- MyST / MyST-NB ----------------------------------------------------------
# Enable handy Markdown features you’re likely to use
myst_enable_extensions = [
    "colon_fence",   # ::: fenced directives
    "deflist",       # definition lists
    "dollarmath",    # $...$ and $$...$$ math (with MathJax)
    # "linkify",     # auto-detect bare links (optional)
    # "substitution" # if you want {substitutions}
]

# Notebook execution settings (MyST-NB)
nb_execution_mode = "off"          # "off" | "auto" | "force" | "cache"
nb_execution_raise_on_error = False

# Make code-output merging clearer (optional niceties)
nb_merge_streams = True
nb_execution_show_tb = True

# Optional: anchor links for Markdown headings (0 = off, or an int depth)
myst_heading_anchors = 3