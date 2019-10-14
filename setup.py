"""Setup script for deepHyperSpec"""

import os.path
from setuptools import setup

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

# This call to setup() does all the work
setup(
    name="deephyp",
    version="0.1.5",
    description="Tools for training and using unsupervised autoencoders and supervised deep learning classifiers for hyperspectral data.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://deephyp.readthedocs.io/en/latest/index.html",
    author="Lloyd Windrim",
    author_email="lloydwindrim@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    packages=["deephyp"],
    include_package_data=True,
    install_requires=[
        "numpy"
    ],
)