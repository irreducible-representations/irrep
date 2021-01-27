
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

from .spacegroup import SpaceGroup
from .bandstructure import BandStructure
from .__aux import str2bool, str2list
from . import __version__ as version


@click.version_option(version)
@click.command(
    help="""

\b
            # ###   ###   #####  ###
            # #  #  #  #  #      #  #
            # ###   ###   ###    ###
            # #  #  #  #  #      #
            # #   # #   # #####  #

\b
version {version}
\b
Calculates the expectation values of symmetry operations 
<Psi_nk | T(g) | Psi_nk >  as well as irreducible representations,
Wannier charge centers (1D) Zak phases and many more.

\b
Examples:
irrep -Ecut=50 -code=abinit -fWFK=Bi_WFK -refUC=0,-1,1,1,0,-1,-1,-1,-1 -kpoints=11 -IBend=5 -kpnames="GM" 
irrep -Ecut=50 -code=espresso -prefix=Bi -refUC=0,-1,1,1,0,-1,-1,-1,-1 -kpoints=11 -IBend=5 -kpnames="GM" 

If you have interest in the code and not sure how to use it, 
do not hesitate to contact the author:

\b
> Stepan S. Tsirkin  
> University of Zurich  
> stepan.tsirkin@physik.uzh.ch
""".format(version=version)
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
    "-plotFile", 
    type=str, 
    help="file where bands for plotting will be written."
    "In development...!"
)
@click.option("-EF", 
    type=float, 
    help="Fermi energy to shift energy-levels. Default: read from DFT output.")
@click.option("-degenThresh", 
    type=float, 
    default=1e-4, 
    help="Threshold to decide whether energy-levels are degenerate. Default: 1e-4")
@click.option(
    "-groupKramers", 
    flag_value=True, 
    default=True, 
    help="Group wave-functions in pairs of Kramers. Default: True."
)
@click.option(
    "-symmetries", 
    type=str, 
    help="Indices of symmetries to be printed. Default: all detected symmetries.")
@click.option(
    "-suffix", 
    type=str, 
    default='tognuplot',
    help="Suffix to name files containing data for band plotting. Default: tognuplot")
def cli(
    ecut,
    fwav,
    fpos,
    fwfk,
    prefix,
    ibstart,
    ibend,
    code,
    spinor,
    kpoints,
    kpnames,
    refuc,
    shiftuc,
    isymsep,
    onlysym,
    zak,
    wcc,
    plotbands,
    plotfile,
    ef,
    degenthresh,
    groupkramers,
    symmetries,
    suffix,
):
    """
    Defines the "irrep" command-line tool interface.
    """
    # TODO: later, this can be split up into separate sub-commands (e.g. for zak, etc.)

    # print("The code was called with the following command-line options:")
    # for k, v in locals().items():
    #     print("{}\t{}".format(k, v))

    # if supplied, convert refUC and shiftUC from comma-separated lists into arrays
    if refuc:
        refuc = np.array(refuc.split(","), dtype=float).reshape((3, 3))
    if shiftuc:
        shiftuc = np.array(shiftuc.split(","), dtype=float).reshape(3)

    # parse input arguments into lists if supplied
    if symmetries:
        symmetries = str2list(symmetries)
    if kpoints:
        kpoints = str2list(kpoints)
    if isymsep:
        isymsep = str2list(isymsep)
    if kpnames:
        kpnames = kpnames.split(",")

    if onlysym:
        spinor = False

    try:
        print(fwfk.split("/")[0].split("-"))
        preline = " ".join(s.split("_")[1] for s in fwfk.split("/")[0].split("-")[:3])
    except Exception as err:
        print(err)
        preline = ""

    if (refuc is not None) and (shiftuc is None):
        shiftuc = np.zeros(3)

    bandstr = BandStructure(
        fWAV=fwav,
        fWFK=fwfk,
        prefix=prefix,
        fPOS=fpos,
        Ecut=ecut,
        IBstart=ibstart,
        IBend=ibend,
        kplist=kpoints,
        spinor=spinor,
        code=code,
        EF=ef,
        onlysym=onlysym,
    )
    bandstr.spacegroup.show(refUC=refuc, shiftUC=shiftuc, symmetries=symmetries)

    if onlysym:
        exit()

    if refuc is None:
        refuc = np.eye(3)
    if shiftuc is None:
        shiftuc = np.zeros(3)

    with open("irreptable-template", "w") as f:
        f.write(bandstr.spacegroup.str(refUC=refuc, shiftUC=shiftuc))

    subbands = {(): bandstr}

    if isymsep is not None:
        for isym in isymsep:
            print("Separating by symmetry operation # ", isym)
            subbands = {
                tuple(list(s_old) + [s_new]): sub
                for s_old, bands in subbands.items()
                for s_new, sub in bands.Separate(
                    isym, degen_thresh=degenthresh, groupKramers=groupkramers
                ).items()
            }

    if zak:
        for k in subbands:
            print("eigenvalue {0}".format(k))
            subbands[k].write_characters(
                degen_thresh=0.001, refUC=refuc, symmetries=symmetries
            )
            print("eigenvalue : #{0} \n Zak phases are : ".format(k))
            zak = subbands[k].zakphase()
            for n, (z, gw, gc, lgw) in enumerate(zip(*zak)):
                print(
                    "   {n:4d}    {z:8.5f} pi gapwidth = {gap:8.4f} gapcenter = {cent:8.3f} localgap= {lg:8.4f}".format(
                        n=n + 1, z=z / np.pi, gap=gw, cent=gc, lg=lgw
                    )
                )

    if wcc:
        for k in subbands:
            print("eigenvalue {0}".format(k))
            # subbands[k].write_characters(degen_thresh=0.001,refUC=refUC,symmetries=symmetries)
            wcc = subbands[k].wcc()
            print(
                "eigenvalue : #{0} \n  WCC are : {1} \n sumWCC={2}".format(
                    k, wcc, np.sum(wcc) % 1
                )
            )

    def short(x, nd=3):
        fmt = "{{0:+.{0}f}}".format(nd)
        if abs(x.imag) < 10 ** (-nd):
            return fmt.format(x.real)
        if abs(x.real) < 10 ** (-nd):
            return fmt.format(x.imag) + "j"
        return short(x.real, nd) + short(1j * x.imag)

    bandstr.write_trace(
        degen_thresh=degenthresh,
        refUC=refuc,
        shiftUC=shiftuc,
        symmetries=symmetries,
    )
    for k, sub in subbands.items():
        if isymsep is not None:
            print(
                "\n\n\n\n ################################################ \n\n\n next subspace:  ",
                " , ".join(
                    "{0}:{1}".format(s, short(ev)) for s, ev in zip(isymsep, k)
                ),
            )
        plotfile=None # being implemented, not finished yet...
        sub.write_characters(
            degen_thresh=degenthresh,
            refUC=refuc,
            shiftUC=shiftuc,
            symmetries=symmetries,
            kpnames=kpnames,
            preline=preline,
            plotFile=plotfile,
        )

    if plotbands:
        for k, sub in subbands.items():
            if isymsep is not None:
                print(
                    "\n\n\n\n ################################################ \n\n\n next subspace:  ",
                    " , ".join(
                        "{0}:{1}".format(s, short(ev)) for s, ev in zip(isymsep, k)
                    ),
                )
                fname = (
                    "bands-"
                    + suffix
                    + "-"
                    + "-".join(
                        "{0}:{1}".format(s, short(ev)) for s, ev in zip(isymsep, k)
                    )
                    + ".dat"
                )
                fname1 = (
                    "bands-sym-"
                    + suffix
                    + "-"
                    + "-".join(
                        "{0}:{1}".format(s, short(ev)) for s, ev in zip(isymsep, k)
                    )
                    + ".dat"
                )
            else:
                fname = "bands-{0}.dat".format(suffix)
                fname1 = "bands-sym-{0}.dat".format(suffix)
            with open(fname, "w") as f:
                f.write(sub.write_bands())
            sub.write_trace_all(degenthresh, fname=fname1)
