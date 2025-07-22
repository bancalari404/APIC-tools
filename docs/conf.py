import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath('../src'))

project = 'pyAPIC'
copyright = '2025, bancalari404'
author = 'bancalari404'
version = '0.1.0'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

# Mock external heavy dependencies
autodoc_mock_imports = ['h5py', 'zernike', 'skimage']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'pyAPIC.tests*']

# Use ReadTheDocs theme (optional)
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
