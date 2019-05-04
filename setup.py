#!/usr/bin/env python

from setuptools import setup

setup(
    # Metadata
    name='FuncionTrees',
    version='0.1',
    description=('Sequence containers for rapidly applying associative'
                 ' operations over continuous ranges'),
    author='CrepeGoat',
    # Contents
    packages=['aggtree.py'],
    # Dependencies
    install_requires='numpy'.split(),
    #tests_requires=['pytest'],
)
