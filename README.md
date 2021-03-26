# IrRep

[![Tests Workflow Status](https://github.com/stepan-tsirkin/irrep/workflows/tests/badge.svg)](https://github.com/stepan-tsirkin/irrep/actions?query=workflow%3Atests)

This is a code to determine symmetry eigenvalues of electronic states obtained by DFT codes, as well as irreducible representations, 
wannier charge centers (1D) and many more

also one can get the trace.txt file for [Check Topological Mat](http://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl) routine 
of [Bilbao Crystallographic Server](https://www.cryst.ehu.es).

Help on usage can be obtained by typing

```
irrep --help
```

Help is under construction and is far from being complete. If you have interest in the code and not sure how to use it, 
feel free to contact the author.

Also, have a look at the examples provided.

An example of using: 

```
irrep -Ecut=50 -code=abinit -fWFK=Bi_WFK -refUC=0,-1,1,1,0,-1,-1,-1,-1  -kpoints=11 -IBend=5  -kpnames="GM"
```


## Installation

Install using `pip`:

```
pip install irrep
```

Currently the code is interfaced VASP, ABINIT and QuantumEspresso, but if interface with other code is needed, please contact te author.

The code relies on [spglib](https://github.com/atztogo/spglib) library to determine the symmetry of the crystal
and the tables of the characters of irreducible representations, obtained from the [Bilbao Crystallographic Server (BCS)](http://www.cryst.ehu.es/)
If you use this code to determine irreps for a scientific publication, please acknowledge BCS and
cite:

> L. Elcoro, B. Bradlyn, Z. Wang, M. G. Vergniory, J. Cano, C. Felser, B. A. Bernevig, D. Orobengoa, G. de la Flor and M. I. Aroyo
"Double crystallographic groups and their representations on the Bilbao Crystallographic Server"
J. of Appl. Cryst. (2017). 50, 1457-1477. doi:10.1107/S1600576717011712

Before releasing this public repository on github on 22th of June 2019, 
the code is mainly written by:

> Stepan S. Tsirkin   
> University of Zurich  

after a fruitful discussion of formalism with Maia G. Vergniory (Donostia International Physics Center/IKERBASQUE, Basque Country, Spain) 

I also acknowledge contributions to the code from Mikel Iraola (Donostia International Physics Center, Basque Country, Spain) 

Further contributions from other authors may be tracked on GitHub [contributors list](https://github.com/stepan-tsirkin/irrep/graphs/contributors). 


## Structure of the package

The files that form the code are organized following a structure that will be described here.

- `irrep`: directory that contains the files that govern the running of the code.
  - `cli.py`: interface to the command line.
  - `__readfiles.py`: routines to read data from DFT output files.
  - `bandstructure.py`: contains the class `BandStructure`, which reads, organizes, treats the data and displays the results.
  - `kpoint.py`: contains the class `Kpoint`, which reads and treats data of a particular k-point and displays results obtained from it.
  - `__gvectors.py`: routines for the generation and transformation of plane-waves.
  - `__spacegroup.py`: classes to read the crystal structure, deduce the space group and deal with symmetry operations.
  - `__aux.py`: auxiliary routines, mainly for type conversion.
  - `__init__.py`: importing version number.
  - `_version.py`: version number.
  - `tests`: directory containing tests for developing purposes.
- `examples`: directory containing input to run examples with different codes and data that has been published in journals, reviews,... In some examples, DFT outputs may not be included due to their large size.
- `tables`: tables of irreducible representations and python scripts for working with them. 
  - `__convertTab.py`: to convert tables of irreducible representations to a user friendly format. (Only for developing)
  - `__init__.py`: classes to read and organize data from tables of irreducible representations.
- `INSTALL`: commands for the installation.
- `LICENSE`: declaration of the license under which the code is made available.
- `setup.py`: routines to install the code.
- `uploadpypi.sh`: to upgrade the code in Pypi. (only for owner's use)

## Development

To develop on `IrRep`, follow these steps:

1. Clone the repository to your local computer, `git clone ...`
2. Ensure you have a modern version of Python installed (3.x+). We recommend 3.6 or higher.
3. Create a new development environment. Either use a virtual environment, for example:
   ```
   python -m venv /path/to/my_irrep_dev_env
   ```
   or, if you're using [anaconda](), a new conda environment.
   ```
   conda create --name my_irrep_dev_env
   ```
4. Activate this virtual environment or conda environment, e.g. by running 
   `source activate /path/to/my_irrep_dev_env/bin/activate` or 
   `conda activate my_irrep_dev_env` as appropriate.
5. Go into the repository directory and run `python setup.py develop`.
6. To run tests you will also need to run `pip install pytest` and then 
   tests can be run by running `pytest`. Currently, tests will run the 
   examples in the `examples` directory and verify their output against a 
   known output.
7. Make changes to the code as required. The `irrep` command line tool 
   will also be available inside this environment.

You can verify your development environment by opening a Python interpretor 
(e.g. running `python`) and running `import irrep` and then `print(irrep.__file__)`.
This should print the path to your local repository containing irrep.
