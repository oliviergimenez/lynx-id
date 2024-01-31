#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="lynx_id",
    version="0.0.1",
    author="PNRIA",
    packages=find_packages(),
    install_requires=[
        "idr_torch>=2.0.0",
        "torch",
        "tqdm",
    ],
)