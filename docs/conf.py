# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
import inspect
import torch  # Or other base classes you want to hide

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------
project = 'Python Research Toolkit - Neural Networks'
copyright = '2024, Gavin Strunk'
author = 'Gavin Strunk'
version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',       # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',   # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.intersphinx',   # Link to other project's documentation (see mapping below)
    'sphinx.ext.viewcode',      # Add a link to the Python source code for classes, functions etc.
    'sphinx.ext.napoleon',      # Allows for Google-style docstrings
    'sphinx.ext.mathjax',       # Formats LaTex code using Jax
    'myst_parser',              # Allows parsing of README.md file
    # 'sphinxcontrib.mermaid',     # Allows for Mermaid diagrams to be rendered
    # 'sphinxcontrib.pseudocode',  # Pseudocode rendering
    # "sphinx_design",            # Enhanced layout design
    "nbsphinx",                 # Rendering jupyter notebooks
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('http://matplotlib.org/', None),
                       'torch': ('https://pytorch.org/docs/stable/', None),
                       }

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autosummary_imported_members = False
autosummary_generate = True         # Turn on sphinx.ext.autosummary
autoclass_content = "both"          # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = True         # Show 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True   # If no docstring, inherit from base class
add_module_names = True             # Keep namespaces in class/method signatures (False)Remove namespaces from class/method signatures
myst_heading_anchors = 2            # Allows implicit linking to header names (X levels) with myst_parser (see https://myst-parser.readthedocs.io/en/latest/syntax/cross-referencing.html)
napoleon_include_init_with_doc = True  # True to list __init___ docstrings separately from the class docstring. False to fall back to Sphinx’s default behavior, which considers the __init___ docstring as part of the class documentation. See 
napoleon_use_admonition_for_examples = False
napoleon_use_ivar = True
# For other napoleon options, see https://sphinxcontrib-napoleon.readthedocs.io/en/latest/sphinxcontrib.napoleon.html

myst_enable_extensions = [
    "colon_fence",  # Enables ::: directives
    "html_admonition",
    "html_image",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Exclusions


# -- Options for HTML output -------------------------------------------------

# Readthedocs theme
# html_theme = "sphinx_rtd_theme"
# html_css_files = ["readthedocs-custom.css"] # Override some CSS settings

# Book Theme
html_theme = "sphinx_book_theme"

# Furo Theme
# html_theme = "furo"
html_logo = "_static/prt-rl-logo.jpg"
html_favicon = "_static/prt-rl-logo.jpg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Hide inherited members from external libraries like torch.nn
def skip_external_inherited_members(app, what, name, obj, skip, options):
    # Always include your own members
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        # Only skip if the object is defined outside your own codebase
        mod = inspect.getmodule(obj)
        if mod:
            # You can adjust this to allow members from your own library
            if mod.__name__.split('.')[0] in {"torch"}:
                return True
    return skip  # Otherwise, keep default behavior

def setup(app):
    app.connect("autodoc-skip-member", skip_external_inherited_members)