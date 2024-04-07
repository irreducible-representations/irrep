
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


import copy
import functools

import numpy as np
import numpy.linalg as la

from .utility import str2bool, BOHR
from .readfiles import ParserAbinit, ParserVasp, ParserEspresso, Hartree_eV
from .readfiles import WAVECARFILE
from .kpoint import Kpoint
from .spacegroup import SpaceGroup
from .gvectors import sortIG, calc_gvectors


class BandStructure:
    """
    Parses files and organizes info about the whole band structure in 
    attributes. Contains methods to calculate and write traces (and irreps), 
    for the separation of the band structure in terms of a symmetry operation 
    and for the calculation of the Zak phase and wannier charge centers.

    Parameters
    ----------
    fWAV : str, default=None
        Name of file containing wave-functions in VASP (WAVECAR format).
    fWFK : str, default=None
        Name of file containing wave-functions in ABINIT (WFK format).
    prefix : str, default=None
        Prefix used for Quantum Espresso calculations or seedname of Wannier90 
        files.
    fPOS : str, default=None
        Name of file containing the crystal structure in VASP (POSCAR format).
    Ecut : float, default=None
        Plane-wave cutoff in eV to consider in the expansion of wave-functions.
    IBstart : int, default=None
        First band to be considered.
    IBend : int, default=None
        Last band to be considered.
    kplist : array, default=None
        List of indices of k-points to be considered.
    spinor : bool, default=None
        `True` if wave functions are spinors, `False` if they are scalars. 
        Mandatory for VASP.
    code : str, default='vasp'
        DFT code used. Set to 'vasp', 'abinit', 'espresso' or 'wannier90'.
    EF : float, default=None
        Fermi-energy.
    onlysym : bool, default=False
        Exit after printing info about space-group.
    spin_channel : str, default=None
        Selection of the spin-channel. 'up' for spin-up, 'dw' for spin-down. 
        Only applied in the interface to Quantum Espresso.
    refUC : array, default=None
        3x3 array describing the transformation of vectors defining the 
        unit cell to the standard setting.
    shiftUC : array, default=None
        Translation taking the origin of the unit cell used in the DFT 
        calculation to that of the standard setting.
    search_cell : bool, default=False
        Whether the transformation to the conventional cell should be computed.
        It is `True` if kpnames was specified in CLI.
    _correct_Ecut0 : float
        In case of VASP, if you get an error like ' computed ncnt=*** != input nplane=*** ', 
        try to set this parameter to a small positive or negative value (usually of order  +- 1e-7)
    trans_thresh : float, default=1e-5
        Threshold to compare translational parts of symmetries.

    Attributes
    ----------
    spacegroup : class
        Instance of `SpaceGroup`.
    spinor : bool
        `True` if wave functions are spinors, `False` if they are scalars. It 
        will be read from DFT files.
    efermi : float
        Fermi-energy. If user set a number as `EF` in CLI, it will be used. If 
        `EF` was set to `auto`, it will try to parse it and set to 0.0 if it 
        could not.
    Ecut0 : float
        Plane-wave cutoff (in eV) used in DFT calulations. Always read from 
        DFT files. Insignificant if `code`='wannier90'.
    Ecut : float
        Plane-wave cutoff (in eV) to consider in the expansion of wave-functions.
        Will be set equal to `Ecut0` if input parameter `Ecut` was not set or 
        the value of this is negative or larger than `Ecut0`.
    Lattice : array, shape=(3,3) 
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    kpoints : list
        Each element is an instance of `class Kpoint` corresponding to a 
        k-point specified in input parameter `kpoints`. If this input was not 
        set, all k-points found in DFT files will be considered.
    _correct_Ecut0 : float
        if you get an error like ' computed ncnt=*** != input nplane=*** ', 
        try to set this parameter to a small positive or negative value (usually of order  +- 1e-7)
    """

    def __init__(
        self,
        fWAV=None,
        fWFK=None,
        prefix=None,
        fPOS=None,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        spinor=None,
        code="vasp",
        EF='0.0',
        onlysym=False,
        spin_channel=None,
        refUC = None,
        shiftUC = None,
        search_cell = False,
        trans_thresh=1e-5,
        _correct_Ecut0 = 0.,
    ):
        code = code.lower()

        if spin_channel is not None:
            spin_channel=spin_channel.lower()
        if spin_channel=='down' : spin_channel='dw'
        
        if code == "vasp":
            self.__init_vasp(
                fWAV, fPOS, Ecut, IBstart, IBend, kplist, spinor, EF=EF, onlysym=onlysym, refUC=refUC, shiftUC=shiftUC, search_cell=search_cell,
                _correct_Ecut0=_correct_Ecut0, trans_thresh=trans_thresh
            )
        elif code == "abinit":
            self.__init_abinit(
                fWFK, Ecut, IBstart, IBend, kplist, EF=EF, onlysym=onlysym, refUC=refUC, shiftUC=shiftUC, search_cell=search_cell, trans_thresh=trans_thresh
            )
        elif code == "espresso":
            self.__init_espresso(
                prefix, Ecut, IBstart, IBend, kplist, EF=EF, onlysym=onlysym, spin_channel=spin_channel, refUC=refUC, shiftUC=shiftUC, search_cell=search_cell, trans_thresh=trans_thresh
            )
        elif code == "wannier90":
            self.__init_wannier(
                prefix, Ecut, IBstart, IBend, kplist, EF=EF, onlysym=onlysym, refUC=refUC, shiftUC=shiftUC, search_cell=search_cell, trans_thresh=trans_thresh
            )
        else:
            raise RuntimeError("Unknown/unsupported code :{}".format(code))

    def __init_vasp(
        self,
        fWAV,
        fPOS,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        spinor=None,
        EF='0.0',
        onlysym=False,
        refUC=None,
        shiftUC=None,
        search_cell=False,
        trans_thresh=1e-5,
        _correct_Ecut0=0.,
    ):
        """
        Initialization for vasp. Read data and save it in attributes.

        Parameters
        ----------
        fWAV : str, default=None
            Filename for wavefunction in VASP WAVECAR format.
        fPOS : str, default=None
            Filename for wavefunction in VASP POSCAR format.
        Ecut : float, default=None
            Plane-wave cutoff in eV to consider in the expansion of wave-functions.
        IBstart : int, default=None
            First band to be considered.
        IBend : int, default=None
            Last band to be considered.
        kplist : , default=None
            List of indices of k-points to be considered.
        spinor : bool, default=None
            `True` if wave functions are spinors, `False` if they are scalars.
        EF : str, default='0.0'
            Fermi-energy.
        onlysym : bool, default=False
            Exit after printing info about space-group.
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=None
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        search_cell : bool, default=False
            Whether the transformation to the conventional cell should be computed.
            It is `True` if kpnames was specified in CLI.
        trans_thresh : float, default=1e-5
            Threshold to compare translational parts of symmetries.
        _correct_Ecut0 : float
            if you get an error like ' computed ncnt=*** != input nplane=*** ', 
            try to set this parameter to a small positive or negative value (usually of order +- 1e-7)
        """
        if spinor is None:
            raise RuntimeError(
                "spinor should be specified in the command line for VASP bandstructure"
            )
        self.spinor = spinor

        # Parse POSCAR and determine space group
        parser = ParserVasp(fPOS, fWAV)
        lattice, positions, typat = parser.parse_poscar()
        self.spacegroup = SpaceGroup(
                              cell=(lattice, positions, typat),
                              spinor=self.spinor,
                              refUC=refUC,
                              shiftUC=shiftUC,
                              search_cell=search_cell,
                              trans_thresh=trans_thresh)
        if onlysym:
            return

        # Fix Fermi level
        if EF.lower() == "auto":
            self.efermi = 0.0
            msg = " (Fermi-energy not found in WAVECAR)"
        else:
            try:
                self.efermi = float(EF)
                msg = ""
            except:
                raise RuntimeError(
                        ("Invalid value for keyword EF. It must be "
                         "a number or 'auto'")
                        )
        print("Efermi = {:.4f} eV".format(self.efermi) + msg)

        NK, NBin, Ecut0, lattice = parser.parse_header()

        # Fix IBstart, IBend and NBout. Move this lines to __init__ at some point
        IBstart = 0 if (IBstart is None or IBstart <= 0) else IBstart - 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")
        if Ecut is None or Ecut > Ecut0 or Ecut <= 0:
            self.Ecut = Ecut0
        else:
            self.Ecut = Ecut
        self.Ecut0 = Ecut0

        self.Lattice = lattice
        # todo: compare lattices parsed from POSCAR and WAVECAR

        self.RecLattice = (
            np.array(
                [
                    np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
                    for i in range(3)
                ]
            )
            * 2
            * np.pi
            / np.linalg.det(self.Lattice)
        )

        print(
            "WAVECAR contains {0} k-points and {1} bands.\n Saving {2} bands starting from {3} in the output".format(
                NK, NBin, NBout, IBstart + 1
            )
        )
        print("Energy cutoff in WAVECAR : ", Ecut0)
        print("Energy cutoff reduced to : ", Ecut)

        if kplist is None:
            kplist = range(NK)
        else:
            kplist -= 1
            kplist = np.array([k for k in kplist if k >= 0 and k < NK])

        # Parse data of k-point from WAVECAR
        self.kpoints = []
        for ik in kplist:
            WF, Energy, kpt, npw = parser.parse_kpoint(ik, NBin, self.spinor)
            # Determine energy of next band from above
            try:
                upper = Energy[IBend]
            except BaseException:
                upper = np.NaN

            # Get rid of bands out of specified window
            Energy = Energy[IBstart:IBend]
            WF = WF[IBstart:IBend]

            # Calculate indices of lattice vectors in plane wave expansion
            kg = calc_gvectors(
                kpt, self.RecLattice, Ecut0, npw, Ecut, spinor=self.spinor
            )

            # Apply plane wave cutoff and sort plane waves based on energy
            if not self.spinor:
                selectG = kg[3]
            else:
                selectG = np.hstack((kg[3], kg[3] + int(npw / 2)))
            WF = WF[:, selectG]

            kp = Kpoint(
                ik=ik,
                NBin=NBin,
                IBstart=IBstart,
                IBend=IBend,
                RecLattice=self.RecLattice,
                symmetries_SG=self.spacegroup.symmetries,
                spinor=self.spinor,
                code="vasp",
                kpt=kpt,
                WF=WF,
                Energy=Energy,
                ig=kg,
                upper=upper
                )
            self.kpoints.append(kp)


    def __init_abinit(
        self,
        WFKname,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        EF='0.0',
        onlysym=False,
        refUC=None,
        shiftUC=None,
        search_cell = False,
        trans_thresh=1e-5
    ):
        """
        Initialization for abinit. Read data and store it in attributes.

        Parameters
        ----------
        WFKname : str
            Filename for wavefunction in ABINIT WFK format.
        Ecut : float, default=None
            Plane-wave cutoff in eV to consider in the expansion of wave-functions.
        IBstart : int, default=None
            First band to be considered.
        IBend : int, default=None
            Last band to be considered.
        kplist : , default=None
            List of indices of k-points to be considered.
        EF : str, default='0.0'
            Fermi-energy.
        onlysym : bool, default=False
            Exit after printing info about space-group.
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=None
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        search_cell : bool, default=False
            Whether the transformation to the conventional cell should be computed.
            It is `True` if kpnames was specified in CLI.
        trans_thresh : float, default=1e-5
            Threshold to compare translational parts of symmetries.
        """

        parser = ParserAbinit(WFKname)
        usepaw = parser.usepaw
        self.spinor = parser.spinor
        self.spacegroup = SpaceGroup(
            cell=(parser.rprimd, parser.xred, parser.typat),
            spinor=self.spinor,
            refUC=refUC,
            shiftUC=shiftUC,
            search_cell=search_cell,
            trans_thresh=trans_thresh
        )
        if onlysym:
            return

        # Set Fermi energy
        if EF.lower() == "auto":
            self.efermi = parser.efermi
        else:
            try:
                self.efermi = float(EF)
            except:
                raise RuntimeError(
                        ("Invalid value for keyword EF. It must be "
                         "a number or 'auto'")
                        )
        print("Efermi: {:.4f} eV".format(self.efermi))

        fWFK = parser.fWFK

        # Set indices and number of bands
        NBin = parser.nband.min()
        NK = parser.nkpt
        IBstart = 0 if (IBstart is None or IBstart <= 0) else IBstart - 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")
        print(
            "WFK contains {0} k-points and {1} bands.\n Saving {2} bands starting from {3} in the output".format(
                NK, NBin, NBout, IBstart + 1
            )
        )

        # Set cutoff to be used to save wave functions
        self.Ecut0 = parser.ecut
        if Ecut is None or Ecut > self.Ecut0 or Ecut <= 0:
            self.Ecut = self.Ecut0
        else:
            self.Ecut = Ecut
        print("Energy cutoff in WFK file : ", self.Ecut0)
        print("Energy cutoff reduced to : ", self.Ecut)

        # Set real and primitive lattice vectors
        self.Lattice = parser.rprimd
        self.RecLattice = (
            np.array(
                [
                    np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
                    for i in range(3)
                ]
            )
            * 2
            * np.pi
            / np.linalg.det(self.Lattice)
        )
        print("lattice vectors:\n", self.Lattice)

        # Set list of indices of k-points to be parsed
        if kplist is None:
            kplist = range(NK)
        else:
            kplist -= 1
            kplist = np.array([k for k in kplist if k >= 0 and k < NK])

        # Parse wave functions of k-points
        self.kpoints = []
        for ik in kplist:
            WF, Energy, kg = parser.parse_kpoint(ik)
            WF = WF[IBstart:IBend,:]
            try:  # Pick energy of IBend+1 band
                upper = Energy[IBend]
            except BaseException:
                upper = np.NaN
            Energy = Energy[IBstart:IBend]
            WF, kg = sortIG(ik, kg, parser.kpt[ik], WF, self.RecLattice, self.Ecut0, self.Ecut, self.spinor)
            kp = Kpoint(
                ik=ik,
                NBin=parser.nband[ik],
                IBstart=IBstart,
                IBend=IBend,
                RecLattice=self.RecLattice,
                symmetries_SG=self.spacegroup.symmetries,
                spinor=self.spinor,
                code="abinit",
                kpt=parser.kpt[ik],
                WF=WF,
                Energy=Energy,
                ig=kg,
                upper=upper
                )
            self.kpoints.append(kp)

            

    def __init_wannier(
        self,
        prefix,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        EF='0.0',
        onlysym=False,
        refUC=None,
        shiftUC=None,
        search_cell = False,
        trans_thresh=1e-5,
    ):
        """
        Initialization for wannier90. Read data and store it in attibutes.

        Parameters
        ----------
        prefix : str, default=None
            Prefix used for Quantum Espresso calculations or seedname of Wannier90 
            files.
        Ecut : float, default=None
            Plane-wave cutoff in eV to consider in the expansion of wave-functions.
        IBstart : int, default=None
            First band to be considered.
        IBend : int, default=None
            Last band to be considered.
        kplist : , default=None
            List of indices of k-points to be considered.
        EF : str, default='0.0'
            Fermi-energy.
        onlysym : bool, default=False
            Exit after printing info about space-group.
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=None
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        search_cell : bool, default=False
            Whether the transformation to the conventional cell should be computed.
            It is `True` if kpnames was specified in CLI.
        trans_thresh : float, default=1e-5
            Threshold to compare translational parts of symmetries.
        """
        if Ecut is None:
            raise RuntimeError("Ecut mandatory for Wannier90")

        fwin = [l.strip().lower() for l in open(prefix + ".win").readlines()]

        def split(l):
            """
            Determine symbol used for assignment and split accordingly.

            Parameters
            ---------
            l : str
                Part of a line read from .win file.
            """
            if "=" in l:
                return l.split("=")
            elif ":" in l:
                return l.split(":")
            else:
                return l.split()

        fwin = [
            [s.strip() for s in split(l)]
            for l in fwin
            if len(l) > 0 and l[0] not in ("!", "#")
        ]
        ind = np.array([l[0] for l in fwin])
        # print(fwin) #com

        def get_param(key, tp, default=None, join=False):
            """
            Return value of a parameter in .win file.

            Parameters
            ----------
            key : str
                Wannier90 input parameter.
            tp : function
                Function to apply to the value of the parameter, before 
                returning it.
            default
                Default value to return in case parameter `key` is not found.
            join : bool, default=False
                If the value of parameter `key` contains more than one element, 
                they will be concatenated with a blank space if `join` is set 
                to `True`. Used when the parameter is `mpgrid`.

            Returns
            -------
            Type(`tp`)
                Return the value of the parameter, after applying function 
                passed es keyword `tp`.

            Raises
            ------
            RuntimeError
                The parameter is not found in .win file, it is found more than 
                once or its value is formed by many elements but it is not
                `mpgrid`.
            """
            i = np.where(ind == key)[0]
            if len(i) == 0:
                if default is None:
                    raise RuntimeError(
                        "parameter {} was not found in {}.win".format(key, prefix)
                    )
                else:
                    return default
            if len(i) > 1:
                raise RuntimeError(
                    "parameter {} was found {} times in {}.win".format(
                        key, len(i), prefix
                    )
                )
            x = fwin[i[0]][1:]  # mp_grid should work
            if len(x) > 1:
                if join:
                    x = " ".join(x)
                else:
                    raise RuntimeError(
                        "length {} found for parameter {}, rather than lenght 1 in {}.win".format(
                            len(x), key, prefix
                        )
                    )
            else:
                x = fwin[i[0]][1]
            return tp(x)

        NBin = get_param("num_bands", int)
        #        print ("nbands=",NBin)
        self.spinor = str2bool(get_param("spinors", str))

        if EF.lower() == "auto":
            try:
                self.efermi = (
                    get_param("fermi_energy", float, 0.0)
                    )
            except:
                print("WARNING : fermi-energy not found. Setting it as zero")
        else:
            try:
                self.efermi = float(EF)
            except:
                raise RuntimeError(
                        ("Invalid value for keyword EF. It must be "
                         "a number or 'auto'")
                        )
        print("Efermi = {:.4f} eV".format(self.efermi))

        NK = np.prod(np.array(get_param("mp_grid", str).split(), dtype=int))

        self.Lattice = None
        kpred = None

        if kplist is None:
            kplist = np.arange(NK) + 1
        else:
            # kplist-=1 #files start from 1 in W90
            kplist = np.array([k for k in kplist if k > 0 and k <= NK])
        found_atoms = False
        # todo : use an iterator to avoid double looping over lines between
        # "begin" and "end"
        iterwin = iter(fwin)

        def check_end(name):
            """
            Check if block in .win file is closed.

            Parameters
            ----------
            name : str
                Name of the block in .win file.
            
            Raises
            ------
            RuntimeError
                Block is not closed.
            """
            s = next(iterwin)
            if " ".join(s) != "end " + name:
                raise RuntimeError(
                    "expected 'end {}, found {}'".format(name, " ".join(s))
                )

        for l in iterwin:
            if l[0].startswith("begin"):
                if l[1] == "unit_cell_cart":
                    if self.Lattice is not None:
                        raise RuntimeError(
                            "'begin unit_cell_cart' found more then once  in {}.win".format(
                                prefix
                            )
                        )
                    j = 0
                    l1 = next(iterwin)
                    if l1[0] in ("bohr", "ang"):
                        units = l1[0]
                        L = [next(iterwin) for i in range(3)]
                    else:
                        units = "ang"
                        L = [l1] + [next(iterwin) for i in range(2)]
                    self.Lattice = np.array(L, dtype=float)
                    if units == "bohr":
                        self.Lattice *= BOHR
                    check_end("unit_cell_cart")
                elif l[1] == "kpoints":
                    if kpred is not None:
                        raise RuntimeError(
                            "'begin kpoints' found more then once  in {}.win".format(
                                prefix
                            )
                        )
                    kpred = np.array(
                        [next(iterwin)[:3] for i in range(NK)], dtype=float
                    )
                    #                    kpred=np.array([kpred[j].split()[:3] for j in kplist],dtype=float)
                    check_end("kpoints")
                elif l[1].startswith("atoms_"):
                    if l[1][6:10] not in ("cart", "frac"):
                        raise RuntimeError("unrecognised block :  '{}' ".format(l[0]))
                    if found_atoms:
                        raise RuntimeError(
                            "'begin atoms_***' found more then once  in {}.win".format(
                                prefix
                            )
                        )
                    found_atoms = True
                    xred = []
                    nameat = []
                    while True:
                        l1 = next(iterwin)
                        if l1[0] == "end":
                            # if l1[1]!=l[0].split()[1]:
                            if l1[1] != l[1]:
                                #                                print (l1[1],l[0].split()[1])
                                raise RuntimeError(
                                    "'{}' ended with 'end {}'".format(
                                        " ".join(l), l1[1]
                                    )
                                )
                            break
                        nameat.append(l1[0])
                        xred.append(l1[1:4])
                    typatdic = {n: i + 1 for i, n in enumerate(set(nameat))}
                    typat = [typatdic[n] for n in nameat]
                    xred = np.array(xred, dtype=float)
                    if l[1][6:10] == "cart":
                        xred = xred.dot(np.linalg.inv(self.Lattice))

        #        print ("lattice vectors:\n",self.Lattice)
        self.RecLattice = (
            np.array(
                [
                    np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
                    for i in range(3)
                ]
            )
            * 2
            * np.pi
            / np.linalg.det(self.Lattice)
        )

        self.spacegroup = SpaceGroup(
            cell=(self.Lattice, xred, typat),
            spinor=self.spinor,
            refUC=refUC,
            shiftUC=shiftUC,
            search_cell=search_cell,
            trans_thresh=trans_thresh
        )
        if onlysym:
            return

        feig = prefix + ".eig"
        eigenval = np.loadtxt(prefix + ".eig")
        try:
            if eigenval.shape[0] != NBin * NK:
                raise RuntimeError("wrong number of entries ")
            ik = np.array(eigenval[:, 1]).reshape(NK, NBin)
            if not np.all(
                ik == np.arange(1, NK + 1)[:, None] * np.ones(NBin, dtype=int)[None, :]
            ):
                raise RuntimeError("wrong k-point indices")
            ib = np.array(eigenval[:, 0]).reshape(NK, NBin)
            if not np.all(
                ib == np.arange(1, NBin + 1)[None, :] * np.ones(NK, dtype=int)[:, None]
            ):
                raise RuntimeError("wrong band indices")
            eigenval = eigenval[:, 2].reshape(NK, NBin)
        except Exception as err:
            raise RuntimeError(" error reading {} : {}".format(feig,err))
  

        IBstart = 0 if (IBstart is None or IBstart <= 0) else IBstart - 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")

        #        print ("eigenvalues are : ",eigenval)
        self.kpoints = [
            Kpoint(
                ik - 1,
                NBin,
                IBstart,
                IBend,
                Ecut,
                None,
                self.RecLattice,
                symmetries_SG=self.spacegroup.symmetries,
                spinor=self.spinor,
                code="wannier",
                eigenval=eigenval[ik - 1],
                kpt=kpred[ik - 1],
            )
            for ik in kplist
        ]

    def __init_espresso(
        self,
        prefix,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        EF='0.0',
        onlysym=False,
        spin_channel=None,
        refUC=None,
        shiftUC=None,
        search_cell = False,
        trans_thresh=1e-5
    ):
        """
        Initialization for Quantum Espresso. Read data and store in attributes.

        Parameters
        ----------
        prefix : str, default=None
            Prefix used for Quantum Espresso calculations or seedname of Wannier90 
            files.
        Ecut : float, default=None
            Plane-wave cutoff in eV to consider in the expansion of wave-functions.
        IBstart : int, default=None
            First band to be considered.
        IBend : int, default=None
            Last band to be considered.
        kplist : , default=None
            List of indices of k-points to be considered.
        EF : str, default='0.0'
            Fermi-energy.
        onlysym : bool, default=False
            Exit after printing info about space-group.
        spin_channel : str, default=None
            Selection of the spin-channel. 'up' for spin-up, 'dw' for spin-down.
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=None
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        search_cell : bool, default=False
            Whether the transformation to the conventional cell should be computed.
            It is `True` if kpnames was specified in CLI.
        trans_thresh : float, default=1e-5
            Threshold to compare translational parts of symmetries.
        """
        import xml.etree.ElementTree as ET  # rm line when moving all parsing to readfiles.py
        parser = ParserEspresso(prefix)
        self.spinor = parser.spinor
        self.Lattice, positions, numbers = parser.parse_lattice()

        # Rm this 5 lines once parser is finished
        mytree = ET.parse(prefix + ".save/data-file-schema.xml")
        myroot = mytree.getroot()
        inp = myroot.find("input")
        outp = myroot.find("output")
        bandstr = outp.find("band_structure")

        self.spacegroup = SpaceGroup(
            cell=(self.Lattice, positions, numbers),
            spinor=self.spinor,
            refUC=refUC,
            shiftUC=shiftUC,
            search_cell=search_cell,
            trans_thresh=trans_thresh
        )
        if onlysym:
            return

        spinpol, Ecut0, EF_in, NK, NBin_list = parser.parse_header()

        IBstartE=0
        if self.spinor and spinpol:
            raise RuntimeError("bandstructure cannot be both noncollinear and spin-polarised. Smth is wrong with the 'data-file-schema.xml'")
        elif spinpol:
            if spin_channel is None:
                raise ValueError("Need to select a spin channel for spin-polarised calculations set  'up' or 'dw'")
            assert (spin_channel in ['dw','up'])
            if spin_channel == 'dw':
                IBstartE = NBin_up[0]
                NBin = NBin_list[1]
            else:
                NBin = NBin_list[0]
        else:
            NBin = NBin_list[0]
            if spin_channel is not None:
                raise ValueError("Found a non-polarized bandstructure, but spin channel is set to {}".format(spin_channel))


        # Set indices for 1st and last bands to be considered
        IBstart = 0 if (IBstart is None or IBstart <= 0) else IBstart - 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")

        # Set cutoff to calculate traces
        if Ecut is None or Ecut > Ecut0 or Ecut <= 0:
            self.Ecut = Ecut0

        self.RecLattice = (
            np.array(
                [
                    np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
                    for i in range(3)
                ]
            )
            * 2
            * np.pi
            / np.linalg.det(self.Lattice)
        )

        # Set Fermi energy
        if EF.lower() == "auto":
            if EF_in is None:
                print("WARNING : fermi-energy not found. Setting it as zero")
                self.efermi = 0.0
            else:
                self.efermi = EF_in
        else:
            try:
                self.efermi = float(EF)
            except:
                raise RuntimeError(
                        ("Invalid value for keyword EF. It must be "
                         "a number or 'auto'")
                        )
        print("Efermi: {:.4f} eV".format(self.efermi))

        # Set list of indices of k-points
        if kplist is None:
            kplist = np.arange(NK)
        else:
            kplist -= 1
            kplist = np.array([k for k in kplist if k >= 0 and k < NK])

        # Parse wave functions
        self.kpoints = []
        kpall = bandstr.findall("ks_energies")
        for ik in kplist:
            WF, Energy, kg, kpt = parser.parse_kpoint(ik, NBin, spin_channel)
            WF = WF[IBstart:IBend,:]
            try:  # Pick energy of IBend+1 band
                upper = Energy[IBend]
            except BaseException:
                upper = np.NaN
            Energy = Energy[IBstart:IBend]
            WF, kg = sortIG(ik+1, kg, kpt, WF, self.RecLattice/2.0, Ecut0, Ecut, self.spinor)
            kp = Kpoint(
                ik,
                NBin,
                IBstart,
                IBend,
                Ecut,
                Ecut0,
                self.RecLattice,
                symmetries_SG=self.spacegroup.symmetries,
                spinor=self.spinor,
                code="espresso",
                spin_channel=spin_channel,
                WF=WF,  # first arg added for abinit (to be kept at the end)
                Energy=Energy,
                ig=kg,
                upper=upper,
                kpt=kpt
            )
            self.kpoints.append(kp)


    def getNK():
        """Getter for `self.kpoints`."""
        return len(self.kpoints)

    NK = property(getNK)

    def write_characters(
        self,
        degen_thresh=1e-8,
        kpnames=None,
        symmetries=None,
        preline="",
        plotFile=None,
    ):
        """
        Calculate irreps, number of band-inversion (if little-group contains 
        inversion), smallest direct gap and indirect gap and print all of them.

        Parameters
        ----------
        degen_thresh : float, default=1e-8
            Threshold energy used to decide whether wave-functions are
            degenerate in energy.
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=np.zeros(3)
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        kpnames : list, default=None
            Labels of maximal k-points at which irreps of bands must be computed. 
            If it is not specified, only traces will be printed, not irreps.
        symmetries : list, default=None
            Index of symmetry operations whose description will be printed. 
        preline : str, default=''
            Characters to write before labels of irreps in file `irreps.dat`.
        plotFile : str, default=None
            Name of file in which energy-levels and corresponding irreps will be 
            written to later place irreps in a band structure plot.

        Returns
        -------
        json_data : `json` object
            Object with output structured in `json` format.
        """
        #        if refUC is not None:
        #        self.spacegroup.show(refUC=refUC,shiftUC=shiftUC)
        #        self.spacegroup.show2(refUC=refUC)
        kpline = self.KPOINTSline()
        json_data = {}
        json_data[ "kpoints_line"] = kpline
        try:
            pFile = open(plotFile, "w")
        except BaseException:
            pFile = None
        NBANDINV = 0
        GAP = np.Inf
        Low = -np.Inf
        Up = np.inf
        json_data["k-points" ] = []
        if kpnames is not None:
            for kpname, KP in zip(kpnames, self.kpoints):
                irreps = self.spacegroup.get_irreps_from_table(kpname, KP.K)
                ninv, low, up , kdata = KP.write_characters(
                    degen_thresh,
                    irreptable=irreps,
                    symmetries=symmetries,
                    preline=preline,
                    efermi=self.efermi,
                    plotFile=pFile,
                    kpl=kpline,
                    symmetries_tables=self.spacegroup.symmetries_tables,
                    refUC=self.spacegroup.refUC,
                    shiftUC=self.spacegroup.shiftUC
                )
                kdata["kpname"] = kpname
                json_data["k-points" ].append(kdata)

                NBANDINV += ninv
                GAP = min(GAP, up - low)
                Up = min(Up, up)
                Low = max(Low, low)
        else:
            for KP, kpl in zip(self.kpoints, kpline):
                ninv, low, up , kdata = KP.write_characters(
                    degen_thresh,
                    symmetries=symmetries,
                    preline=preline,
                    efermi=self.efermi,
                    plotFile=pFile,
                    kpl=kpl,
                    symmetries_tables=self.spacegroup.symmetries_tables,
                    refUC=self.spacegroup.refUC,
                    shiftUC=self.spacegroup.shiftUC
                )
                kdata["kp in line"] = kpl
                json_data["k-points" ].append(kdata)
                NBANDINV += ninv
                GAP = min(GAP, up - low)
                Up = min(Up, up)
                Low = max(Low, low)

        if self.spinor:
            print(
                "number of inversion-odd Kramers pairs IN THE LISTED KPOINTS: ",
                int(NBANDINV / 2),
                "  Z4= ",
                int(NBANDINV / 2) % 4,
            )
            json_data["number of inversion-odd Kramers pairs"]  = int(NBANDINV / 2)
            json_data["Z4"] = int(NBANDINV / 2) % 4,
        else:
            print("number of inversion-odd states : ", NBANDINV)
            json_data["number of inversion-odd states"]  = NBANDINV

        #        print ("Total number of inversion-odd Kramers pairs IN THE LISTED KPOINTS: ",NBANDINV,"  Z4= ",NBANDINV%4)
        print("Minimal direct gap:", GAP, " eV")
        print("indirect  gap:", Up - Low, " eV")
        json_data["indirect gap (eV)"] =  Up-Low
        json_data["Minimal direct gap (eV)"] =  GAP
       
        return json_data

    def getNbands(self):
        """
        Return number of bands (if equal over all k-points), raise RuntimeError 
        otherwise.

        Returns
        -------
        int
            Number of bands in every k-point.
        """
        nbarray = [k.Nband for k in self.kpoints]
        if len(set(nbarray)) > 1:
            raise RuntimeError(
                "the numbers of bands differs over k-points:{0} \n cannot write trace.txt \n".format(
                    nbarray
                )
            )
        if len(nbarray) == 0:
            raise RuntimeError(
                "do we have any k-points??? NB={0} \n cannot write trace.txt \n".format(
                    nbarray
                )
            )
        return nbarray[0]

    def write_trace(
        self,
        degen_thresh=1e-8,
        kpnames=None,
        symmetries=None,
    ):
        """
        Generate `trace.txt` file to upload to the program `CheckTopologicalMat` 
        in `BCS <https://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl>`_ .

        Parameters
        ----------
        degen_thresh : float, default=1e-8
            Threshold energy used to decide whether wave-functions are
            degenerate in energy.
        kpnames : list, default=None
            Labels of maximal k-points at which irreps of bands must be computed. 
            If it is not specified, only traces will be printed, not irreps.
        symmetries : list, default=None
            Index of symmetry operations whose description will be printed. 
        """
        f = open("trace.txt", "w")
        f.write(
            (
                " {0}  \n"
                + " {1}  \n"  # Number of bands below the Fermi level  # Spin-orbit coupling. No: 0, Yes: 1
            ).format(self.getNbands(), 1 if self.spinor else 0)
        )

        f.write(
                self.spacegroup.write_trace()
                )
        # Number of maximal k-vectors in the space group. In the next files
        # introduce the components of the maximal k-vectors))
        f.write("  {0}  \n".format(len(self.kpoints)))
        for KP in self.kpoints:
            f.write(
                "   ".join(
                    "{0:10.6f}".format(x)
                    for x in KP.K
                )
                + "\n"
            )
        for KP in self.kpoints:
            f.write(
                KP.write_trace(degen_thresh, symmetries=symmetries, efermi=self.efermi)
            )

    def Separate(self, isymop, degen_thresh=1e-5, groupKramers=True):
        """
        Separate band structure according to the eigenvalues of a symmetry 
        operation.
        
        Parameters
        ----------
        isymop : int
            Index of symmetry used for the separation.
        degen_thresh : float, default=1e-5
            Energy threshold used to determine degeneracy of energy-levels.
        groupKramers : bool, default=True
            If `True`, states will be coupled by Kramers' pairs.

        Returns
        -------
        subspaces : dict
            Each key is an eigenvalue of the symmetry operation and the
            corresponding value is an instance of `class` `BandStructure` for 
            the subspace of that eigenvalue.
        """
        if isymop == 1:
            return {1: self}
        symop = self.spacegroup.symmetries[isymop - 1]
        #print("Separating by symmetry operation # ", isymop)
        symop.show()
        kpseparated = [
            kp.Separate(symop, degen_thresh=degen_thresh, groupKramers=groupKramers)
            for kp in self.kpoints
        ] # each element is a dict with separated bandstructure of a k-point
        allvalues = np.array(sum((list(kps.keys()) for kps in kpseparated), []))
        #        print (allvalues)
        #        for kps in kpseparated :
        #            allvalues=allvalues | set( kps.keys())
        #        allvalues=np.array(allavalues)
        if groupKramers:
            allvalues = allvalues[np.argsort(np.real(allvalues))].real
            borders = np.hstack(
                (
                    [0],
                    np.where(abs(allvalues[1:] - allvalues[:-1]) > 0.01)[0] + 1,
                    [len(allvalues)],
                )
            )
            #            nv=len(allvalues)
            if len(borders) > 2:
                allvalues = set(
                    [allvalues[b1:b2].mean() for b1, b2 in zip(borders, borders[1:])]
                ) # unrepeated Re parts of all eigenvalues
                subspaces = {}
                for v in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(v - vk))]
                        if abs(vk0 - v) < 0.05:
                            other.kpoints.append(K[vk0])
                    subspaces[v] = other # unnecessary indent ?
                return subspaces
            else:
                return dict({allvalues.mean(): self})
        else:
            allvalues = allvalues[np.argsort(np.angle(allvalues))]
            print("allvalues:", allvalues)
            borders = np.where(abs(allvalues - np.roll(allvalues, 1)) > 0.01)[0]
            nv = len(allvalues)
            if len(borders) > 0:
                allvalues = set(
                    [
                        np.roll(allvalues, -b1)[: (b2 - b1) % nv].mean()
                        for b1, b2 in zip(borders, np.roll(borders, -1))
                    ]
                )
                print("distinct values:", allvalues)
                subspaces = {}
                for v in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(v - vk))]
                        #                    print ("v,vk",v,vk)
                        #                    print ("v,vk",v,vk[np.argmin(np.abs(v-vk))])
                        if abs(vk0 - v) < 0.05:
                            other.kpoints.append(K[vk0])
                        subspaces[v] = other
                return subspaces
            else:
                return dict({allvalues.mean(): self})

    def zakphase(self):
        """
        Calculate Zak phases along a path for a set of states.

        Returns
        -------
        z : array
            `z[i]` contains the total  (trace) zak phase (divided by 
            :math:`2\pi`) for the subspace of the first i-bands.
        array
            The :math:`i^{th}` element is the gap between :math:`i^{th}` and
            :math:`(i+1)^{th}` bands in the considered set of bands. 
        array
            The :math:`i^{th}` element is the mean energy between :math:`i^{th}` 
            and :math:`(i+1)^{th}` bands in the considered set of bands. 
        array
            Each line contains the local gaps between pairs of bands in a 
            k-point of the path. The :math:`i^{th}` column is the local gap 
            between :math:`i^{th}` and :math:`(i+1)^{th}` bands.
        """
        overlaps = [
            x.overlap(y)
            for x, y in zip(self.kpoints, self.kpoints[1:] + [self.kpoints[0]])
        ]
        print("overlaps")
        for O in overlaps:
            print(np.abs(O[0, 0]), np.angle(O[0, 0]))
        print("   sum  ", np.sum(np.angle(O[0, 0]) for O in overlaps) / np.pi)
        #        overlaps.append(self.kpoints[-1].overlap(self.kpoints[0],g=np.array( (self.kpoints[-1].K-self.kpoints[0].K).round(),dtype=int )  )  )
        nmax = np.min([o.shape for o in overlaps])
        # calculate zak phase in incresing dimension of the subspace (1 band,
        # 2 bands, 3 bands,...)
        z = np.angle(
            [[la.det(O[:n, :n]) for n in range(1, nmax + 1)] for O in overlaps]
        ).sum(axis=0) % (2 * np.pi)
        #        print (np.array([k.Energy[1:] for k in self.kpoints] ))
        #        print (np.min([k.Energy[1:] for k in self.kpoints],axis=0) )
        emin = np.hstack(
            (np.min([k.Energy[1:nmax] for k in self.kpoints], axis=0), [np.Inf])
        )
        emax = np.max([k.Energy[:nmax] for k in self.kpoints], axis=0)
        locgap = np.hstack(
            (
                np.min(
                    [k.Energy[1:nmax] - k.Energy[0 : nmax - 1] for k in self.kpoints],
                    axis=0,
                ),
                [np.Inf],
            )
        )
        return z, emin - emax, (emin + emax) / 2, locgap

    def wcc(self):
        """
        Calculate Wilson loops.

        Returns
        -------
        array
            Eigenvalues of the Wilson loop operator, divided by :math:`2\pi`.

        """
        overlaps = [
            x.overlap(y)
            for x, y in zip(self.kpoints, self.kpoints[1:] + [self.kpoints[0]])
        ]
        nmax = np.min([o.shape for o in overlaps])
        wilson = functools.reduce(
            np.dot,
            [functools.reduce(np.dot, np.linalg.svd(O)[0:3:2]) for O in overlaps],
        )
        return np.sort((np.angle(np.linalg.eig(wilson)) / (2 * np.pi)) % 1)

    def write_bands(self, locs=None):
        """
        Generate lines for a band structure plot, with cummulative length of the
        k-path as values for the x-axis and energy-levels for the y-axis.

        Returns
        -------
        str
            Lines to write into a file that will be parsed to plot the band 
            structure.
        """
        #        print (locs)
        kpline = self.KPOINTSline()
        nbmax = max(k.Nband for k in self.kpoints)
        EN = np.zeros((nbmax, len(kpline)))
        EN[:, :] = np.Inf
        for i, k in enumerate(self.kpoints):
            EN[: k.Nband, i] = k.Energy - self.efermi
        if locs is not None:
            loc = np.zeros((nbmax, len(kpline), len(locs)))
            for i, k in enumerate(self.kpoints):
                loc[: k.Nband, i, :] = k.getloc(locs).T
            return "\n\n\n".join(
                "\n".join(
                    (
                        "{0:8.4f}   {1:8.4f}  ".format(k, e)
                        + "  ".join("{0:8.4f}".format(l) for l in L)
                    )
                    for k, e, L in zip(kpline, E, LC)
                )
                for E, LC in zip(EN, loc)
            )
        else:
            return "\n\n\n".join(
                "\n".join(
                    ("{0:8.4f}   {1:8.4f}  ".format(k, e)) for k, e in zip(kpline, E)
                )
                for E in EN
            )

    def write_trace_all(
        self,
        degen_thresh=1e-8,
        symmetries=None,
        fname="trace_all.dat",
    ):
        """
        Write in a file the description of symmetry operations, energy-levels 
        and irreps calculated in all k-points.

        Parameters
        ----------
        degen_thresh : float, default=1e-8
            Threshold energy used to decide whether wave-functions are
            degenerate in energy.
        symmetries : list, default=None
            Index of symmetry operations whose traces will be printed. 
        fname : str, default=trace_all.dat
            Name of output file.
        """
        f = open(fname, "w")
        kpline = self.KPOINTSline()

        f.write(
            (
                "# {0}  # Number of bands below the Fermi level\n"
                + "# {1}  # Spin-orbit coupling. No: 0, Yes: 1\n"  #
            ).format(self.getNbands(), 1 if self.spinor else 0)
        )
        # add lines describing symmetry operations
        f.write(
            "\n".join(
                ("#" + l)
                for l in self.spacegroup.write_trace().split("\n")
            )
            + "\n\n"
        )
        for KP, KPL in zip(self.kpoints, kpline):
            f.write(
                KP.write_trace_all(
                    degen_thresh, symmetries=symmetries, efermi=self.efermi, kpline=KPL
                )
            )


    def KPOINTSline(self, kpred=None, breakTHRESH=0.1):
        """
        Calculate cumulative length along a path in reciprocal space.

        Parameters
        ----------
        kpred : list, default=None
            Each element contains the direct coordinates of a k-point in the
            attribute `kpoints`.
        breakTHRESH : float, default=0.1
            If the distance between two neighboring k-points in the path is 
            larger than `breakTHRESH`, it is taken to be 0.

        Returns
        -------
        array
            Each element is the cumulative distance along the path up to a 
            k-point. The first element is 0, so that the number of elements
            matches the number of k-points in the path.
        """
        if kpred is None:
            kpred = [k.K for k in self.kpoints]
        KPcart = np.array(kpred).dot(self.RecLattice)
        K = np.zeros(KPcart.shape[0])
        k = np.linalg.norm(KPcart[1:, :] - KPcart[:-1, :], axis=1)
        k[k > breakTHRESH] = 0.0
        K[1:] = np.cumsum(k)
        return K
