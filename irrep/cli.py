
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
##  Written by Stepan Tsirkin                                    #
##  e-mail: stepan.tsirkin@ehu.eus                               #
##################################################################

"""
Defines the command line interface to "irrep".
"""

import numpy as np

import click
from monty.serialization import dumpfn, loadfn

from .bandstructure import BandStructure
from .utility import sort_vectors, str2list, short, log_message
from . import __version__ as version


class LoadContextFromConfig(click.Command):
    """
    Custom class to allow supplying command context from an
    input file. Thanks to https://stackoverflow.com/a/46391887
    for concept.
    """

    def invoke(self, ctx):

        config_file = ctx.params["config"]

        if config_file is not None:

            # load from either a .yml or .json
            config_data = loadfn(config_file)

            # sanitize inputs to be all lower-case
            config_data = {k.lower(): v for k, v in config_data.items()}

            for param, value in ctx.params.items():
                if param in config_data:
                    ctx.params[param] = config_data[param]

        return super(LoadContextFromConfig, self).invoke(ctx)


@click.version_option(version)
@click.command(
    cls=LoadContextFromConfig,
    help=f"""

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
> stepan.tsirkin@epfl.ch
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
    "-correct_Ecut0",
    type=float,
    default=0.,
    help="In case of VASP, if you get an error like ' computed ncnt=*** != input nplane=*** ', "
    "try to set this parameter to a small positive or negative value (usually of order  +- 1e-7)"
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
    "-gpaw_calc",
    type=str,
    help="Filename for gpaw calculator. "
    'Only used if code is "gpaw".',
)
@click.option(
    "-prefix",
    type=str,
    help="Prefix used for Quantum Espresso calculations (data should be in prefix.save) or seedname of Wannier90 files. ",
)
@click.option(
    "-from_sym_file",
    type=str,
    help="if present, the symmetry operations will be read from this file, instead of those computed by spglib",
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
    type=click.Choice(["vasp", "abinit", "espresso", "wannier90", "gpaw"]),
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
    default=None,
    help="Comma-separated list of k-point names (as in the tables) with one entry per each "
    "value in the k-points list. Important! K-points is assumed to be an ordered list!",
)
@click.option(
    "-refUC",
    type=str,
    default=None,
    help="The lattice vectors of the reference unit cell (as given in the crystallographic tables) "
    "expressed in terms of the unit cell vectors used in the calculation. "
    "Nine comma-separated numbers.",
)
@click.option(
    "-shiftUC",
    type=str,
    default=None,
    help="The vector to shift the calculated unit cell origin (in units of the calculated lattice), "
    "to get the unit cell as defined in crystallographic tables. Three comma-separated numbers.",
)
@click.option(
    "-isymsep",
    help="Index of the symmetry to separate the eigenstates. "
    "with new method works with any code/pseudopotential"
    "Previously worked well only for norm-conserving potentials.",
)
@click.option(
    "-onlysym",
    flag_value=True,
    default=False,
    help="Only calculate the symmetry operations",
)
@click.option(
    "-unk_formatted",
    flag_value=True,
    default=False,
    help="expect UNK files to be formatted (only relevant when -code=wannier90 )",
)
@click.option(
    "-writesym",
    flag_value=True,
    default=False,
    help="write the prefix.sym file needed for the Wannier90 sitesym calculations",
)
@click.option(
    "-alat",
    type=float,
    default=None,
    help="for writesym - the alat parameter. For QE, it is read from the prefix.save/data-file-schema.xml"
    "For other codes needs to be provided. (To be honest, the .sym file is useless for other codes for now, but still ..)",
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
@click.option("-EF",
    type=str,
    default='0.0',
    help=("Fermi energy to shift energy-levels. Default: 0.0. If it is"
          " set to a number, this value will be used to shift the "
          "energy-levels. If it is set to 'auto', the code will try "
          "to parse it from DFT files and set it to 0.0 if it could "
          "not do so."
          )
)
@click.option("-degenThresh",
    type=float,
    default=1e-4,
    help="Threshold to decide whether energy-levels are degenerate. Default: 1e-4")
@click.option("-symprec",
    type=float,
    default=1e-5,
    help="Symmetry precision. Default: 1e-5. Passed to spglib get_symmetry"
)
@click.option(
    "-groupKramers",
    flag_value=True,
    default=False,
    help="Group wave-functions in pairs of Kramers. Default: False."
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
@click.option("-config", type=click.Path(),
              help="Define irrep inputs from a configuration file in YAML or JSON format.")
@click.option(
    "-searchcell",
    flag_value=True,
    default=False,
    help=("Find transformation to conventional unit cell. If it is "
          "not specified, the transformation to the conventional "
          "unit cell will not be calculated and, if refUC or shiftUC "
          "were given, it will not be checked if they correctly lead "
          "to the conventional cell."
          "Default: False, unless -kpnames was specified in CLI"
          )
)
@click.option("-trans_thresh",
    type=float,
    default=1e-5,
    help=("Threshold to compare translational parts of symmetries."
          "Default: 1e-5"
          )
)
@click.option("-magmom",
    type=str,
    default=None,
    help=("Path to magnetic moments' file. One row per atom, three "
          "coordinates in Cartesian.")
)
@click.option("-v",
              count=True,
              default=1,
              help=("Verbosity flag. -vv: very detailed info, recommended "
                    "when you get an error. -v (default for CLI): info about "
                    "some decissions taken internaly through the code's "
                    "execution, recommended when the code runs without "
                    "errors, but the result is not what you expected. If you "
                    "don't set this tag, you will get the basic info.")
)
@click.option("-json_file",
              type=str,
              default="irrep-output.json",
              help="File to save the output in JSON format. (without "
              "extension, the '.json' will be added automatically)"
)
@click.option(
    "--time-reversal",
    flag_value=True,
    default=False,
    help=("Consider TRS a symmetry and use the corresponding gray group.")
)
@click.option("--print-hs-kpoints",
              flag_value=True,
              default=False,
              help="Print high-symmetry k-points in the calculation and reference cell."
)
@click.option("--symmetry-indicators",
              flag_value=True,
              default=False,
              help="Compute symmetry indicators if they are non-trivial. "
              "Irreps must be identified in the process."
)
@click.option("--ebr-decomposition",
              flag_value=True,
              default=False,
              help="Compute the EBR decomposition and topological classification "
              "according to TQC. Irreps must be identified in the process."
)
def cli(
    ecut,
    fwav,
    fpos,
    fwfk,
    gpaw_calc,
    prefix,
    ibstart,
    ibend,
    code,
    spinor,
    kpoints,
    kpnames,
    refuc,
    shiftuc,
    symprec,
    isymsep,
    onlysym,
    writesym,
    alat,
    from_sym_file,
    zak,
    wcc,
    plotbands,
    ef,
    degenthresh,
    groupkramers,
    symmetries,
    suffix,
    config,
    searchcell,
    correct_ecut0,
    trans_thresh,
    magmom,
    time_reversal,
    v,
    json_file,
    unk_formatted,
    print_hs_kpoints,
    symmetry_indicators,
    ebr_decomposition
):
    """
    Defines the "irrep" command-line tool interface.
    """
    # TODO: later, this can be split up into separate sub-commands (e.g. for zak, etc.)

    # if supplied, convert refUC and shiftUC from comma-separated lists into arrays
    if refuc:
        refuc = np.array(refuc.split(","), dtype=float).reshape((3, 3))
    if shiftuc:
        shiftuc = np.array(shiftuc.split(","), dtype=float).reshape(3)

    # rename the v flag to verbosity, to avoid overlap with local variables
    verbosity = v
    # Warning about kpnames
    if kpnames is not None:
        searchcell = True

    elif not searchcell:
        log_message("Warning: transformation to the convenctional unit "
                    "cell will not be calculated, nor its validity checked "
                    "(if given). See the description of flag -searchcell "
                    "on:\n"
                    "irrep --help",
                    verbosity, 1)
        log_message("Warning: -kpnames not specified. Only traces of "
                    "symmetry operations will be calculated. Remember that "
                    "-kpnames must be specified to identify irreps",
                    verbosity, 1)

    # parse input arguments into lists if supplied
    if symmetries:
        symmetries = str2list(symmetries)
    if kpoints:
        kpoints = str2list(kpoints)
    if isymsep:
        isymsep = str2list(isymsep)
    if kpnames:
        kpnames = kpnames.split(",")

    # Decide if wave functions should be kept in memory after calculating trace
    if isymsep or wcc or zak:
        save_wf = True
    else:
        save_wf = False

    # Read magnetic moments from a file
    if magmom is not None:
        try:
            magnetic_moments = np.loadtxt(magmom)
            if magnetic_moments.ndim == 1:
                magnetic_moments = np.expand_dims(magnetic_moments, axis=0)
        except FileNotFoundError:
            print(f"The magnetic moments file was not found: {magmom}")
        except ValueError:
            print(f"Error reading magnetic moments file: {magmom}")
    elif time_reversal:
        magnetic_moments = True
        print("WARNING: --time-reversal set without providing magnetic moments "
              "via --magmom. Magnetic grey groups will be used. Results are "
              "still valid.")
    elif symmetry_indicators:
        magnetic_moments = True
        print("WARNING:\nusing --symmetry-indicators with a non-magnetic "
              "crystal invariant under time-reversal. Switching to the "
              "grey group with time-reversal. Results are still valid.")
    else:
        magnetic_moments = None

    if print_hs_kpoints:  # don't read wave functions, only identify SG
        onlysym = True

    bandstr = BandStructure(
        fWAV=fwav,
        fWFK=fwfk,
        calculator_gpaw=gpaw_calc,
        prefix=prefix,
        fPOS=fpos,
        Ecut=ecut,
        IBstart=ibstart,
        IBend=ibend,
        kplist=kpoints,
        spinor=spinor,
        calculate_traces=True,
        code=code,
        EF=ef,
        onlysym=onlysym,
        refUC=refuc,
        shiftUC=shiftuc,
        search_cell=searchcell,
        degen_thresh=degenthresh,
        magmom=magnetic_moments,
        save_wf=save_wf,
        symprec=symprec,
        unk_formatted=unk_formatted,
        verbosity=verbosity,
        from_sym_file=from_sym_file,
        include_TR=(magnetic_moments is not None),  # if magnetic, include TR
        irreps=True,  # always identify irreps when called from irrep code
    )

    bandstr.spacegroup.show()

    if print_hs_kpoints:
        bandstr.spacegroup.print_hs_kpoints()

    if writesym:
        bandstr.spacegroup.write_sym_file(filename=prefix + ".sym", alat=alat)

    if onlysym:
        json_data = {}
        json_data["spacegroup"] = bandstr.spacegroup.json(symmetries=symmetries)
        dumpfn(json_data, json_file, indent=4)
        exit()

    with open("irreptable-template", "w") as f:
        f.write(
            bandstr.spacegroup.str()
        )

    # Identify irreps. If kpnames wasn't set, all will be labelled as None
    bandstr.identify_irreps(kpnames, verbosity=verbosity)

    # Temporary, until we make it valid for isymsep
    bandstr.write_characters()

    if ebr_decomposition:
        bandstr.compute_ebr_decomposition()
        bandstr.print_ebr_decomposition()

    if symmetry_indicators:
        bandstr.print_symmetry_indicators()

    # Write irreps.dat file
    if kpnames is not None:
        bandstr.write_irrepsfile()

    # Write trace.txt file
    bandstr.write_trace()

    # Temporary, until we make it valid for isymsep
    json_data = {}
    json_data["spacegroup"] = bandstr.spacegroup.json(symmetries=symmetries)
    json_bandstr = bandstr.json()
    json_data['characters and irreps'] = [{"subspace": json_bandstr}]

    # Separate in terms of symmetry eigenvalues
    subbands = {(): bandstr}

    if isymsep is not None:
        json_data["separated by symmetry"] = True
        json_data["separating symmetries"] = isymsep
        tmp_subbands = {}
        for isym in isymsep:
            print(f"\n-------- SEPARATING BY SYMMETRY # {isym} --------")
            for s_old, bs in subbands.items():
                separated = bs.Separate(isym, groupKramers=groupkramers, verbosity=verbosity)
                for s_new, bs_separated in separated.items():
                    # Later, the sorting of evals will be based on the
                    # argument of each eval in terms of [0,2pi). So it's
                    # convenient to round the imag part around 0 to avoid
                    # pulling apart 1.0+j1e-4 and 1.0-j1e-4
                    if abs(s_new.imag) < 1e-4:
                        s_new = complex(s_new.real, 0.0)
                    evals = tuple(list(s_old) + [s_new])
                    tmp_subbands[evals] = bs_separated
            subbands = tmp_subbands
        json_data["characters and irreps"] = []

        # sort to have consistency between runs
        for k in sort_vectors(subbands.keys()):
            sub = subbands[k]
            if isymsep is not None:
                print(
                    "\n\n\n\n ################################################ \n\n\n NEXT SUBSPACE:  ",
                    " , ".join(f"sym # {s} -> eigenvalue {short(ev)}" for s, ev in zip(isymsep, k)),
                )
                sub.write_characters()
                json_data["characters and irreps"].append({"symmetry eigenvalues": np.array(k),
                                                           "subspace": sub.json(symmetries)})
    else:
        json_data["separated by symmetry"] = False


    dumpfn(json_data, json_file, indent=4)

    if zak:
        for k in subbands:
            print(f"symmetry eigenvalue : {k} \n Traces are : ")
            subbands[k].write_characters()
            print(f"symmetry eigenvalue : {k} \n Zak phases are : ")
            zak = subbands[k].zakphase()
            for n, (z, gw, gc, lgw) in enumerate(zip(*zak)):
                print(f"   {n + 1:4d}    {z / np.pi:8.5f} pi gapwidth = {gw:8.4f}"
                      f" gapcenter = {gc:8.3f} localgap= {lgw:8.4f}")
            exit()

    if wcc:
        for k in subbands:
            print(f"symmetry eigenvalue {k}")
            # subbands[k].write_characters(degen_thresh=0.001,refUC=refUC,symmetries=symmetries)
            wcc_val = subbands[k].wcc()
            print(f"symmetry eigenvalue : {k} \n  WCC are : {wcc_val} \n sumWCC={np.sum(wcc_val) % 1}")
        exit()

    if plotbands:
        print("\nplotbands = True --> writing bands")
        for k, sub in subbands.items():
            if isymsep is not None:
                print("\n\n\n\n ################################################ \n\n\n next subspace:  ",
                    " , ".join(f"{s}:{short(ev)}" for s, ev in zip(isymsep, k)), )
                fname = (
                    "bands-" + suffix + "-" +
                    "-".join(f"{s}:{short(ev)}" for s, ev in zip(isymsep, k)) +
                    ".dat")
            else:
                fname = f"bands-{suffix}.dat"
            sub.write_plotfile(fname)
