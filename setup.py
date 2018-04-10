#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
from setuptools import setup

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__HEMCEE_SETUP__ = True
import hemcee  # NOQA

setup(
    name="hemcee",
    version=hemcee.__version__,
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/hemcee",
    license="MIT",
    packages=["hemcee"],
    install_requires=["numpy", "scipy", "tqdm"],
    description="",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst", "LICENSE", "CITATION"]},
    include_package_data=True,
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache 2 License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=True,
)
