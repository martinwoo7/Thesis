from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="WCDS",
    version="0.1.0",
    author="Martin",
    description="WiSARD for Clustering Data Streams",
    long_description=long_description,
    license="MIT",
    long_description_content_type="text/markdown",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "WCDS"},
    packages=find_packages(where="WCDS"),
    python_requires=">=3.6",
    install_requires=[],
)