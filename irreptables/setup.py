import setuptools

long_description = """
Tables of irreducible representations from the Bilbao Crystallographic Server, 
and an interface to irrep code". Refer to 
https://pypi.org/project/irrep/
https://github.com/stepan-tsirkin/irrep"""


print ("setting irreptables package")
setuptools.setup(
    name="irreptables",
    author="Stepan S. Tsirkin",
    author_email="stepan.tsirkin@ehu.eus",
    description="Tables of characters of irreducible representations for double space groups. Part of 'irrep' package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
                "numpy",
                "scipy>=1.0",
                "spglib>=1.14",
                "Click"
                    ],
    include_package_data=True,
    url="https://github.com/stepan-tsirkin/irrep",
    packages=["irreptables"],   #setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ])

