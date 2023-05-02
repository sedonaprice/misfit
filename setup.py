#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import re

import logging

try:
    from setuptools import setup
except:
    from distutils.core import setup

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'misfit', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)

logging.basicConfig()
log = logging.getLogger(__file__)


with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['numpy', 'scipy', 'matplotlib', 'astropy',
                'emcee', 'corner', 'dill',
                'h5py', 'pandas', 'six']

setup_requirements = ['numpy']

setup_args = {'name': 'misfit',
        'author': "Sedona Price",
        'author_email': 'sedona.price@gmail.com',
        'python_requires': '>=3.6',
        'classifiers': [
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: 3-clause BSD',
            'Natural Language :: English',
            "Topic :: Scientific/Engineering",
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
        'description': "Misaligned kinematic fitting of galaxies from slit spectra",
        'install_requires': requirements,
        'setup_requires': setup_requirements,
        'license': "3-clause BSD",
        'long_description': readme,
        'include_package_data': True,
        'packages': ['misfit', 'misfit.fit', 'misfit.general',
                     'misfit.mock', 'misfit.model', 'misfit.plot'],
        'package_data': {'misfit': ['lib/*']},
        'url': 'http://github.com/sedonaprice/misfit', 
        'version': __version__ }


# Add CONDA include and lib paths if necessary
conda_include_path = "."
conda_lib_path = "."
if 'CONDA_PREFIX' in os.environ:
    conda_include_path = os.path.join(os.getenv('CONDA_PREFIX'), 'include')
    conda_lib_path = os.path.join(os.getenv('CONDA_PREFIX'), 'lib')
    log.debug('conda_include_path: {!r}'.format(conda_include_path))
    log.debug('conda_lib_path: {!r}'.format(conda_lib_path))


setup( **setup_args)
log.info("Installation successful!")