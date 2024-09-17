#!/usr/bin/env python

"""The setup script."""

import io
from os import path as op
from setuptools import setup, find_packages



requirements = []

setup_requirements = []

test_requirements = []

setup(
    author="52013141240",
    author_email="2833980897@qq.com",
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="A Python package for installing commonly used packages for geospatial analysis and data visualization with only one command.",
    install_requires=install_requires,
    dependency_links=dependency_links,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="geospatial",
    name="geospatial",
    packages=find_packages(include=["geospatial", "geospatial.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/5201314240/unet",
    zip_safe=False,
)
