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
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../../src/slog'))


# -- Project information -----------------------------------------------------

project = 'SLOG'
copyright = '2021, Jad Sadek'
author = 'Jad Sadek'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

def skip(app, what, name, obj, skip, options):
    if name.startswith("_"):
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip)

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.


extensions = [ 'sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.graphviz', 'sphinx.ext.inheritance_diagram', 'numpydoc']

numpydoc_attributes_as_param_list= True
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # python
    'sequence': ':term:`python:sequence`',
    'iterable': ':term:`python:iterable`',
    'string': 'str',
    # numpy
    'array': 'numpy.ndarray',
    'dtype': 'numpy.dtype',
    'ndarray': 'numpy.ndarray',
    'array-like': ':term:`numpy:array_like`',
    'array_like': ':term:`numpy:array_like`',
    #plurals
}

napoleon_type_aliases = numpydoc_xref_aliases
napoleon_use_rtype = False
napoleon_use_admonition_for_notes = True

intersphinx_mapping = {'python': ('http://docs.python.org/3', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('http://matplotlib.sourceforge.net/', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


#graphviz_dot = shutil.which('dot')
graphviz_output_format = 'svg'

#autodoc_class_signature = 'separated'

add_module_names = False
add_function_parentheses = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'css/customs.css'
]
html_logo = "_static/SLOG.png"
