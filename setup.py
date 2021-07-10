import setuptools, sys

with open("README.md", "r") as fh:
    long_description = fh.read()

import irrep

setuptools.setup(
    name="irrep",
    version=irrep.__version__,
    author="Stepan S. Tsirkin",
    author_email="stepan.tsirkin@uzh.ch",
    description="A tool to get symmetry proberties of ab-initio wavefunctions, irreduible representations and more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "scipy>=1.0",
        "spglib>=1.14",
        "lazy_property",
        "Click",
        "monty",
        "ruamel.yaml",
        "irreptables",
    ],
    include_package_data=False,
    url="https://github.com/stepan-tsirkin/irrep",
    packages=["irrep"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points="""
        [console_scripts]
        irrep=irrep.cli:cli
    """,
)
