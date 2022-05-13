# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "mirgecom"
copyright = ("2020, University of Illinois Board of Trustees")
author = "Center for Exascale-Enabled Scramjet Design at the University of Illinois"

_ver_file = "../mirgecom/version.py"
with open(_ver_file) as ver_file:
    ver_src = ver_file.read()

ver_dic = {}
exec(compile(ver_src, _ver_file, "exec"), ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])

# The full version, including alpha/beta/rc tags.
release = ver_dic["VERSION_TEXT"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx_copybutton",
    ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/pyopencl/": None,
    "https://documen.tician.de/modepy/": None,
    "arraycontext": ("https://documen.tician.de/arraycontext/", None),
    "https://documen.tician.de/meshmode/": None,
    "https://documen.tician.de/grudge/": None,
    "https://documen.tician.de/pytato/": None,
    "https://documen.tician.de/loopy/": None,
    "https://documen.tician.de/dagrt/": None,
    "https://documen.tician.de/leap/": None,
    "https://documen.tician.de/pymbolic/": None,
    "https://documen.tician.de/pytools/": None,
    "https://pyrometheus.readthedocs.io/en/latest": None,
    "https://logpyle.readthedocs.io/en/latest/": None,
    "https://psutil.readthedocs.io/en/latest/": None,
    "https://mpi4py.readthedocs.io/en/stable/": None,
    }

autoclass_content = "class"

todo_include_todos = True

nitpicky = True

mathjax3_config = {
    "tex2jax": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

rst_prolog = """
.. |mirgecom| replace:: *MIRGE-Com*
"""
