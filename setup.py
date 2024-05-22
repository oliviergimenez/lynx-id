#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open('requirements.txt') as req_fd:
    INSTALL_REQUIRES = req_fd.read().splitlines()

setup(
    name="lynx_id",
    version="0.0.1",
    author="PNRIA",
    packages=find_packages(),
    package_data={
        'lynx_id': [
            'ressources/*',  # add anything here in the future
            'ressources/models/*',  # add models here in the future
            'ressources/configs/*',  # add configurations here in the future
            'ressources/tests/data_test.txt',  # test data file
        ],
    },
    install_requires=INSTALL_REQUIRES
)
