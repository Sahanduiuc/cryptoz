import sys
import os
from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'requirements.txt')) as f:
    install_reqs = [r.rstrip() for r in f.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

with open("cryptoz/__version__.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open("classifiers.txt") as f:
    classifiers = f.read().strip().split('\n')

setup(name='cryptoz',
      version=version,
      description='Lightweight Python library for tracking cryptocurrency markets',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Oleg Polakow',
      url='https://github.com/polakowo/cryptoz',
      packages=find_packages(),
      install_requires=install_reqs,
      keywords='binance cryptocurrency statistics visualization bitcoin ethereum btc eth altcoins analysis trading',
      python_requires='>=3.*',
      classifiers=classifiers)