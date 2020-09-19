
            # ###   ###   #####  ###
            # #  #  #  #  #      #  #
            # ###   ###   ###    ###
            # #  #  #  #  #      #
            # #   # #   # #####  #


##################################################################
## This file is distributed as part of                           #
## "IrRep" code and under terms of GNU General Public license v3 #
## see LICENSE file in the                                       #
##                                                               #
##  Written by Stepan Tsirkin, University of Zurich.             #
##  e-mail: stepan.tsirkin@physik.uzh.ch                         #
##################################################################

"""
Defines the command line interface to "irrep".
"""

import sys
import numpy as np
import datetime
import math
import click

from .__spacegroup import SpaceGroup
from . import __bandstructure as BS
from .__aux import str2bool, str2list

from irrep import __version__ as version


@click.version_option(version)
@click.command(
    help="""

\b
            # ###   ###   #####  ###
            # #  #  #  #  #      #  #
            # ###   ###   ###    ###
            # #  #  #  #  #      #
            # #   # #   # #####  #

Calculates the expectation values of symmetry operations 
<Psi_nk | T(g) | Psi_nk >  as well as irreducible representations,
Wannier charge centers (1D) Zak phases and many more.

Examples:
irrep -Ecut=50 -code=abinit -fWFK=Bi_WFK -refUC=0,-1,1,1,0,-1,-1,-1,-1 -kpoints=11 -IBend=5 -kpnames="GM" 
irrep -Ecut=50 -code=espresso -prefix=Bi -refUC=0,-1,1,1,0,-1,-1,-1,-1 -kpoints=11 -IBend=5 -kpnames="GM" 

If you have interest in the code and not sure how to use it, 
do not hesitate to contact the author:

\b
> Stepan S. Tsirkin  
> University of Zurich  
> stepan.tsirkin@physik.uzh.ch
"""
)
@click.option(
    "-Ecut",
    type=float,
    help="Energy cut-off in eV used in the calculation. "
    "A value of 50 eV is recommended. If not set, will default to "
    "the cut-off used in the DFT calculation.",
)
@click.option(
    "-fWAV",
    type=str,
    default="WAVECAR",
    help="Filename for wavefunction in VASP WAVECAR format. "
    'Only used if code is "vasp".',
)
@click.option(
    "-fPOS",
    type=str,
    default="POSCAR",
    help="Filename for wavefunction in VASP POSCAR format. "
    'Only used if code is "vasp".',
)
@click.option(
    "-fWFK",
    type=str,
    help="Filename for wavefunction in ABINIT WFK format. "
    'Only used if code is "abinit".',
)
@click.option(
    "-prefix",
    type=str,
    help="Prefix for QuantumEspresso calculations (data should be in prefix.save). "
    'Only used if code is "espresso".',
)
@click.option(
    "-IBstart",
    type=int,
    default=0,
    help="The first band to be considered. "
    "If <= 0 starting from the lowest band (count from one).",
)
@click.option(
    "-IBend",
    type=int,
    default=0,
    help="The last band to be considered. "
    "If <=0 up to  the highest band (count from one).",
)
@click.option(
    "-code",
    type=click.Choice(["vasp", "abinit", "espresso", "wannier90"]),
    default="vasp",
    help="Set which electronic structure code to interface with. If using ABINIT, always use "
    '"istwfk=1".',
)
@click.option(
    "-spinor",
    flag_value=True,
    default=False,
    help="Indicate the wavefunctions are spinor. Only used " 'if code is "vasp".',
)
@click.option(
    "-kpoints",
    type=str,
    help="Comma-separated list of k-point indices (starting from 1).",
)
@click.option(
    "-kpnames",
    type=str,
    help="Comma-separated list of k-point names (as in the tables) with one entry per each "
    "value in the k-points list. Important! K-points is assumed to be an ordered list!",
)
@click.option(
    "-refUC",
    type=str,
    help="The lattice vectors of the reference unit cell (as given in the crystallographic tables) "
    "expressed in terms of the unit cell vectors used in the calculation. "
    "Nine comma-separated numbers.",
)
@click.option(
    "-shiftUC",
    type=str,
    help="The vector to shift the calculated unit cell origin (in units of the calculated lattice), "
    "to get the unit cell as defined in crystallographic tables. Three comma-separated numbers.",
)
@click.option(
    "-NB",
    default=0,
    help="Number of bands in the output. If NB <= 0 all bands are used.",
)
@click.option(
    "-isymsep",
    help="Index of the symmetry to separate the eigenstates. Works well only for norm-conserving "
    "potentials as in ABINIT.",
)
@click.option(
    "-onlysym",
    flag_value=True,
    default=False,
    help="Only calculate the symmetry operations",
)
@click.option("-ZAK", flag_value=True, default=False, help="Calculate Zak phase")
@click.option(
    "-WCC", flag_value=True, default=False, help="Calculate Wannier charge centres"
)
@click.option(
    "-plotbands",
    flag_value=True,
    default=False,
    help="Write gnuplottable files with all symmetry eigenvalues",
)
@click.option(
    "-WCC", flag_value=True, default=False, help="Calculate Wannier charge centres"
)
def cli(*args, **kwargs):
    """
    Defines the "irrep" command-line tool interface.
    """
    # TODO: later, this can be split up into separate sub-commands
    pass
