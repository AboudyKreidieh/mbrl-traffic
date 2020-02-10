#!/usr/bin/env python3
# flake8: noqa
"""Setup script for the mbrl-traffic repository."""
from os.path import dirname, realpath
from setuptools import find_packages, setup
from mbrl_traffic.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='mbrl-traffic',
    version=__version__,
    packages=find_packages(),
    install_requires=_read_requirements_file(),
    description='mbrl-traffic: Macroscopic model-based reinforcement learning '
                'framework for control of mixed-autonomy traffic',
    author='Aboudy Kreidieh',
    url='https://github.com/AboudyKreidieh/mbrl-traffic',
    author_email='aboudy@berkeley.edu',
    zip_safe=False,
)
