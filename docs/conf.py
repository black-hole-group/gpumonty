# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GPUmonty'
copyright = '2025, Pedro Naethe Motta'
author = 'Pedro Naethe Motta'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# Add Breathe to extensions
extensions = [
    'breathe',
    'sphinx_rtd_theme',
]

# Tell Breathe where the Doxygen XML files are
# (Assuming Doxygen output is in a folder named 'xml' inside your project root)
breathe_projects = {"GPUmonty": "../xml/"}
breathe_default_project = "GPUmonty"

# Set the theme to the professional "Read the Docs" style
html_theme = "sphinx_rtd_theme"

# Tell Sphinx to ignore CUDA-specific keywords
cpp_id_attributes = [
    "__host__",
    "__device__",
    "__global__",
    "__forceinline__",
]


html_theme_options = {
    # 'True' hides the plus signs for non-active sections.
    # 'False' forces the sidebar to show expansion icons for everything.
    'collapse_navigation': False,
    
    # This determines how many levels deep the sidebar will go.
    'navigation_depth': 4,
    
    # Optional: If you want the [+] signs to only show titles 
    # and not every single sub-function until you click.
    'titles_only': False 
}

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# Force Breathe to show the #define values
breathe_show_define_initializer = True