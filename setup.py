## python3 setup.py sdist
## python3 setup.py sdist bdist_wheel
## python -m twine upload *
import setuptools

with open("README", "r") as fh:
    long_description = fh.read()


from irrep  import __version__ as version

setuptools.setup(
     name='irrep',  
     version=version,
     author="Stepan S. Tsirkin",
     author_email="stepan.tsirkin@uzh.ch",
     description="a tool to get symmetry proberties of ab-initio wavefunctions, irreduible representations and more",
     long_description=long_description,
     long_description_content_type="text/markdown",
     install_requires=['numpy', 'scipy >= 1.0', 'spglib >=1.14' ],
     include_package_data=True,
     url="https://github.com/stepan-tsirkin/irrep",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
         "Operating System :: OS Independent",
     ],
 )