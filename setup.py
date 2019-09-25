import sys
import os
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Package version
VERSION = (HERE / "cryptoz" / "__version__.py").read_text().strip().split('\n')[-1].split()[-1].strip("\"'")

# The text of the README file
README = (HERE / "README.md").read_text()

# Install requirements
INSTALL_REGS = [
    r.rstrip() for r in (HERE / 'requirements.txt').read_text().strip().split('\n') 
    if not r.startswith('#') and not r.startswith('git+')
]

# Classifier strings
CLASSIFIERS = (HERE / "classifiers.txt").read_text().strip().split('\n')

setup(name='cryptoz',
      version=VERSION,
      description='Lightweight Python library for tracking cryptocurrency markets',
      long_description=README,
      long_description_content_type="text/markdown",
      author='Oleg Polakow',
      author_email='olegpolakow@gmail.com',
      url='https://github.com/polakowo/cryptoz',
      packages=find_packages(exclude=("tests",)),
      install_requires=INSTALL_REGS,
      keywords='binance cryptocurrency statistics visualization bitcoin ethereum btc eth altcoins analysis trading',
      python_requires='>=3.6',
      classifiers=CLASSIFIERS)