#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

from setuptools import find_packages, setup, Command
from shutil import rmtree
import sys
import os
import io
from pathlib import Path

# Package meta-data.
NAME = 'simplenn'
URL="https://github.com/facundoq/simplenn"


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = "No README.md in your MANIFEST.in"

# Load the package's __version__.py module as a dictionary.
VERSION = "0.1.0"


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        
        dist_path=Path(here)/'dist'
        if dist_path.exists():
            self.status('Removing previous builds…')
            rmtree(dist_path)


        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description='A single library to (down)load all existing sign language handshape datasets. This library allows you to automatically download and load various sign language handshape datasets. Currently supporting 13 different datasets.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Facundo Quiroga',
    author_email='fquiroga@lidi.info.unlp.edu.ar',
    python_requires='>=3.6.0',
    url=URL,
    project_urls={
        "Bug Tracker": URL+"/issues",
        "Documentation": URL,
        "Source Code": URL,
    },
    #packages=find_packages(exclude=('simplenn','samples')),
    # If your package is a single module, use this instead of 'packages':
    py_modules=['simplenn'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires= [ 'numpy','matplotlib','tqdm','colorama','pydataset'],
    include_package_data=True,
    license='GNU Affero General Public License v3 or later (AGPLv3+)',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: PyPy',
        # 'Programming Language :: Python :: 3 :: Only',
        "License :: OSI Approved :: Python Software Foundation License",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
