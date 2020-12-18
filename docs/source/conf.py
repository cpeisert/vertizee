# Sphinx configuration file for the documentation builder.
#
# For a full list of Sphinx options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import vertizee.version

# -- Project information -----------------------------------------------------
project = "Vertizee"
copyright = "2020, The Vertizee Authors"
author = "The Vertizee Authors"

# The short X.Y.Z version.
version = vertizee.version.__version__

_release_candidate = ""  # RC format: -rc.#   Example: '-rc.0'
# The full version, including alpha/beta/rc tags.
release = f"{version}{_release_candidate}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",  # Jupyter Notebook Tools for Sphinx
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.imgconverter",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",  # Converts Google-style docstrings to reStructuredText
    "sphinx.ext.viewcode",
    "recommonmark",  # Markdown support
]

# https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
autosectionlabel_prefix_document = True

# autodoc options
autodoc_typehints = "signature"

autodoc_default_options = {
    "autoclass_content": "class",
    "member-order": "bysource",
    "members": True,
    "show-inheritance": True,
    "special-members": "__call__, __getitem__, __len__",
    "undoc-members": True,
}

mathjax_config = {
    "extensions": ["tex2jax.js"],
    "jax": ["input/TeX", "output/HTML-CSS"],
}

latex_elements = {"extrapackages": r"\usepackage{ latexsym }"}

# Napoleon settings
napoleon_google_docstring = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_param = False  # Show the Parameter Types in separate line.
napoleon_use_rtype = True  # Show the Return Type in separate line.

# nbsphinx options
html_sourcelink_suffix = ""
nbsphinx_kernel_name = "python3"
# nbsphinx_input_prompt = 'In [%s]:'
# nbsphinx_output_prompt = 'Out [%s]:'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".ipynb", ".md"]

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = 'alabaster'

html_theme_options = {
    "canonical_url": "https://cpeisert.github.io/vertizee/documentation/stable/",
    "navigation_depth": 3,
    "logo_only": True,
}

# html_logo = "_static/vertizee.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app) -> None:
    app.add_css_file("css/code-snippets.css")
