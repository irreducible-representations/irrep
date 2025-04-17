
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


import copy
import functools
import os
import json

import numpy as np
from functools import cached_property

from .readfiles import ParserAbinit, ParserVasp, ParserEspresso, ParserW90, ParserGPAW
from .kpoint import Kpoint
from .spacegroup import SpaceGroupIrreps
from .gvectors import sortIG, calc_gvectors, symm_matrix
from .utility import get_block_indices, grid_from_kpoints, log_message, UniqueListMod1

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
    calculator_gpaw : GPAW, default=None
        Instance of GPAW calculator. Mandatory for GPAW.
    fPOS : str, default=None
        Name of file containing the crystal structure in VASP (POSCAR format).
    Ecut : float, default=None
        Plane-wave cutoff in eV to consider in the expansion of wave-functions.
        mandatory for GPAW and Wannier90.
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
        DFT code used. Set to 'vasp', 'abinit', 'espresso' or 'wannier90' or 'gpaw'.
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
    trans_thresh : float, default=1e-5
        Threshold to compare translational parts of symmetries.
    degen_thresh : float, default=1e-8
        Threshold to determine the degeneracy of energy levels.
    calculate_traces : bool
        If `True`, traces of symmetries will be calculated. Useful to icreate 
        instances of `BandStructure` faster.
    save_wf : bool
        Whether wave functions should be kept as attribute after calculating 
        traces.
    verbosity : int, default=0
        Number controlling the verbosity. 
        0: minimalistic printing. 
        1: print info about decisions taken internally by the code, recommended 
        when the code runs without errors but the result is not the expected.
        2: print detailed info, recommended when the code stops with an error
    magmom : array(num_atoms, 3)
        Magnetic moments of atoms in the unit cell. 
    include_TR : bool
        If `True`, the symmetries involving time-reversal will be included in the spacegroup.
        if magmom is None and include_TR is True, the magnetic moments will be set to zero (non-magnetic calculation with TR)
    unk_formatted : bool
        If `True`, the input files are expected to be formatted text files. If False, the input files are expected to be binary files.

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
        set, all k-points found in DFT files will be considered. ACCESSED 
        DIRECTLY BY BANDUPPY>=0.3.4. DO NOT CHANGE UNLESS NECESSARY. NOTIFY 
        THE DEVELOPERS IF ANY CHANGES ARE MADE.
    num_bandinvs : int
        Property that returns the number of inversion odd states in the 
        given TRIM.
    gap_direct : float
        Property that returns the smallest direct gap in the given k points.
    gap_indirect : float
        Property that returns the smallest indirect gap in the given k points.
    num_k : int
        Property that returns the number of k points in the attribute `kpoints`
    num_bands : int
        Property that returns the number of bands. Used to write trace.txt.
    """

    def __init__(
        self,
        fWAV=None,
        fWFK=None,
        prefix=None,
        calculator_gpaw=None,
        fPOS=None,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        spinor=None,
        code="vasp",
        calculate_traces=False,
        EF='0.0',
        onlysym=False,
        spin_channel=None,
        refUC = None,
        shiftUC = None,
        search_cell = False,
        trans_thresh=1e-5,
        degen_thresh=1e-8,
        save_wf=True,
        verbosity=0,
        alat=None,
        from_sym_file=None,
        normalize=True,
        magmom=None,
        include_TR=False,
        unk_formatted=False,
    ):

        code = code.lower()
        if spin_channel is not None:
            spin_channel=spin_channel.lower()
        if spin_channel=='down':
            spin_channel='dw'
        
        if code == "vasp":

            if spinor is None:
                raise RuntimeError(
                    "spinor should be specified in the command line for VASP bandstructure"
                )
            self.spinor = spinor
            parser = ParserVasp(fPOS, fWAV, onlysym)
            self.Lattice, positions, typat = parser.parse_poscar(verbosity)
            if not onlysym:
                NK, NBin, self.Ecut0, lattice = parser.parse_header()
                if not np.allclose(self.Lattice, lattice):
                    raise RuntimeError("POSCAR and WAVECAR contain different lattices")
                EF_in = None  # not written in WAVECAR

        elif code == "abinit":

            parser = ParserAbinit(fWFK)
            (nband,
             NK,
             self.Lattice,
             self.Ecut0,
             self.spinor,
             typat,
             positions,
             EF_in) = parser.parse_header(verbosity=verbosity)
            NBin = max(nband)

        elif code == "espresso":

            parser = ParserEspresso(prefix)
            self.spinor = parser.spinor
            # alat is saved to be used to write the prefix.sym file
            self.Lattice, positions, typat, _alat = parser.parse_lattice()
            if alat is None:
                alat = _alat
            spinpol, self.Ecut0, EF_in, NK, NBin_list = parser.parse_header()

            # Set NBin
            if self.spinor and spinpol:
                raise RuntimeError("bandstructure cannot be both noncollinear and spin-polarised. Smth is wrong with the 'data-file-schema.xml'")
            elif spinpol:
                if spin_channel is None:
                    raise ValueError("Need to select a spin channel for spin-polarised calculations set  'up' or 'dw'")
                assert (spin_channel in ['dw','up'])
                if spin_channel == 'dw':
                    NBin = NBin_list[1]
                else:
                    NBin = NBin_list[0]
            else:
                NBin = NBin_list[0]
                if spin_channel is not None:
                    raise ValueError("Found a non-polarized bandstructure, but spin channel is set to {}".format(spin_channel))

        elif code == "wannier90":

            if Ecut is None:
                raise RuntimeError("Ecut mandatory for Wannier90")

            self.Ecut0 = Ecut
            parser = ParserW90(prefix, unk_formatted=unk_formatted)
            NK, NBin, self.spinor, EF_in = parser.parse_header()
            self.Lattice, positions, typat, kpred = parser.parse_lattice()
            Energies = parser.parse_energies()
        elif code == "gpaw":
            parser = ParserGPAW(calculator=calculator_gpaw,
                                spinor=False if spinor is None else spinor)
            (NBin,
             kpred,
             self.Lattice,
             self.spinor,
             typat,
             positions,
             EF_in) = parser.parse_header()
            if Ecut is None:
                raise RuntimeError("Ecut mandatory for GPAW")
            self.Ecut0 = Ecut
            NK = kpred.shape[0]
        else:
            raise RuntimeError(f"Unknown/unsupported code :{code}")

        cell = (self.Lattice, positions, typat)

        self.spacegroup = SpaceGroupIrreps(
                              cell=cell,
                              spinor=self.spinor,
                              refUC=refUC,
                              shiftUC=shiftUC,
                              search_cell=search_cell,
                              trans_thresh=trans_thresh,
                              verbosity=verbosity,
                              alat=alat,
                              from_sym_file=from_sym_file,
                              magmom=magmom,
                              include_TR=include_TR,
                              )
        self.magnetic = self.spacegroup.magnetic

        if onlysym:
            return

        # Set Fermi energy
        if EF.lower() == "auto":
            if EF_in is None:
                self.efermi = 0.0
                msg = "WARNING : fermi-energy not found. Setting it as 0 eV"
                log_message(msg, verbosity, 1)
            else:
                self.efermi = EF_in
        else:
            try:
                self.efermi = float(EF)
            except:
                raise RuntimeError("Invalid value for keyword EF. It must be "
                                   "a number or 'auto'")

        log_message(f"Efermi: {self.efermi:.4f} eV", verbosity, 1)

        # Fix indices of bands to be considered
        if IBstart is None or IBstart <= 0:
            IBstart = 0
        else:
            IBstart -= 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")

        # Set cutoff to calculate traces
        if Ecut is None or Ecut > self.Ecut0 or Ecut <= 0:
            self.Ecut = self.Ecut0
        else:
            self.Ecut = Ecut

        # Calculate vectors of reciprocal lattice
        self.RecLattice = np.zeros((3,3), dtype=float)
        for i in range(3):
            self.RecLattice[i] = np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
        self.RecLattice *= (2.0*np.pi/np.linalg.det(self.Lattice))

        # To do: create writer of description for this class
        msg = ("WAVECAR contains {} k-points and {} bands.\n"
               "Saving {} bands starting from {} in the output"
               .format(NK, NBin, NBout, IBstart + 1))
        log_message(msg, verbosity, 1)
        msg = f"Energy cutoff in WAVECAR : {self.Ecut0}"
        log_message(msg, verbosity, 1)
        msg = f"Energy cutoff reduced to : {self.Ecut}"
        log_message(msg, verbosity, 1)

        # Create list of indices for k-points
        if kplist is None:
            kplist = range(NK)
        else:
            kplist -= 1
            kplist = np.array([k for k in kplist if k >= 0 and k < NK])

        # Parse wave functions at each k-point
        self.kpoints = []
        for ik in kplist:

            if code == 'vasp':
                msg = f'Parsing wave functions at k-point #{ik:>3d}'
                log_message(msg, verbosity, 2)
                WF, Energy, kpt, npw = parser.parse_kpoint(ik, NBin, self.spinor)
                kg = calc_gvectors(kpt,
                                   self.RecLattice,
                                   self.Ecut0,
                                   npw,
                                   self.Ecut,
                                   spinor=self.spinor,
                                   verbosity=verbosity
                                   )
                if not self.spinor:
                    selectG = kg[3]
                else:
                    selectG = np.hstack((kg[3], kg[3] + int(npw / 2)))
                WF = WF[:, selectG]

            elif code == 'abinit':
                NBin = parser.nband[ik]
                kpt = parser.kpt[ik]
                msg = f'Parsing wave functions at k-point #{ik:>3d}: {kpt}'
                log_message(msg, verbosity, 2)
                WF, Energy, kg = parser.parse_kpoint(ik)
                WF, kg = sortIG(ik, kg, kpt, WF, self.RecLattice, self.Ecut0, self.Ecut, self.spinor, verbosity=verbosity)

            elif code == 'espresso':
                msg = f'Parsing wave functions at k-point #{ik:>3d}'
                log_message(msg, verbosity, 2)
                WF, Energy, kg, kpt = parser.parse_kpoint(ik, NBin, spin_channel, verbosity=verbosity)
                WF, kg = sortIG(ik+1, kg, kpt, WF, self.RecLattice/2.0, self.Ecut0, self.Ecut, self.spinor, verbosity=verbosity)

            elif code == 'wannier90':
                kpt = kpred[ik]
                Energy = Energies[ik]
                ngx, ngy, ngz = parser.parse_grid(ik+1)
                kg = calc_gvectors(kpred[ik],
                                   self.RecLattice,
                                   self.Ecut,
                                   spinor=self.spinor,
                                   nplanemax=np.max([ngx, ngy, ngz]) // 2,
                                   verbosity=verbosity
                                   )
                selectG = tuple(kg[0:3])
                msg = f'Parsing wave functions at k-point #{ik:>3d}: {kpt}'
                log_message(msg, verbosity, 2)
                WF = parser.parse_kpoint(ik+1, selectG)
            elif code == 'gpaw':
                kpt = kpred[ik]
                Energy, WF, kg, kpt= parser.parse_kpoint(ik,
                                                 RecLattice=self.RecLattice,
                                                 Ecut=self.Ecut)
            

            # Pick energy of IBend+1 band to calculate gaps
            try:
                upper = Energy[IBend] - self.efermi
            except BaseException:
                upper = np.nan

            # Preserve only bands in between IBstart and IBend
            WF = WF[IBstart:IBend]
            Energy = Energy[IBstart:IBend] - self.efermi

            # saved to further use in Separate()
            self.kwargs_kpoint=dict(
                degen_thresh=degen_thresh,
                refUC=self.spacegroup.refUC,
                shiftUC=self.spacegroup.shiftUC,
                symmetries_tables=self.spacegroup.u_symmetries_tables,
                save_wf=save_wf,
                verbosity=verbosity,
                calculate_traces=calculate_traces,
                )

            kp = Kpoint(
                ik=ik,
                kpt=kpt,
                WF=WF,
                Energy=Energy,
                ig=kg,
                upper=upper,
                num_bands=NBout,
                RecLattice=self.RecLattice,
                symmetries_SG=self.spacegroup.u_symmetries,
                spinor=self.spinor,
                kwargs_kpoint=self.kwargs_kpoint,
                normalize=normalize,
                )
            self.kpoints.append(kp)
        del WF

    @property 
    def lattice(self):
        return self.Lattice

    @property
    def num_k(self):
        '''Getter for the number of k points'''
        return len(self.kpoints)

    def identify_irreps(self, kpnames, verbosity=0):
        '''
        Identifies the irreducible representations of wave functions based on 
        the traces of symmetries of the little co-group. Each element of 
        `kpoints`  will be assigned the attribute `irreps` with the labels of 
        irreps.

        Parameters
        ----------
        kpnames : list
            List of labels of the maximal k points.
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing
        '''

        for ik, KP in enumerate(self.kpoints):
            
            if kpnames is not None:
                irreps = self.spacegroup.get_irreps_from_table(kpnames[ik], KP.k, verbosity=verbosity)
            else:
                irreps = None
            KP.identify_irreps(irreptable=irreps)

    def write_characters(self):
        '''
        For each k point, write the energy levels, their degeneracies, traces 
        of the little cogroup's symmetries, and the direct and indirect gaps. 
        Also the irreps, if they have been identified. If the crystal is 
        inversion symmetries, the number of total inversion odd states, the 
        Z2 and Z4 numbers will be written.
        '''

        for KP in self.kpoints:

            # Print block of irreps and their characters
            KP.write_characters()

            # Print number of inversion odd Kramers pairs
            if KP.num_bandinvs is None:
                print("\nInvariant under inversion: No")
            else:
                print("\nInvariant under inversion: Yes")
                if self.spinor and not self.magnetic:
                    print("Number of inversions-odd Kramers pairs : {}"
                          .format(int(KP.num_bandinvs / 2))
                          )
                else:
                    print("Number of inversions-odd states : {}"
                          .format(KP.num_bandinvs))

            # Print gap with respect to next band
            if not np.isnan(KP.upper):
                print("Gap with upper bands: ", KP.upper - KP.Energy_mean[-1])
        
        # Print total number of band inversions
        if self.spinor and not self.magnetic:
            print("\nTOTAL number of inversions-odd Kramers pairs : {}"
                  .format(int(self.num_bandinvs/2)))
        else:
            print("TOTAL number of inversions-odd states : {}"
                  .format(self.num_bandinvs))
        
        if not self.magnetic:
            print('Z2 invariant: {}'.format(int(self.num_bandinvs/2 % 2)))
            print('Z4 invariant: {}'.format(int(self.num_bandinvs/2 % 4)))

        # Print indirect gap and smalles direct gap
        print('Indirect gap: {}'.format(self.gap_indirect))
        print('Smallest direct gap in the given set of k-points: {}'.format(self.gap_direct))
    

    def json(self, kpnames=None):
        '''
        Prepare a dictionary to save the data in JSON format.

        Parameters
        ----------
        kpnames : list
            List of labels of the maximal k points.

        Returns
        -------
        json_data : dict
            Dictionary with the data.
        '''

        kpline = self.KPOINTSline()
        json_data = {}
        json_data['kpoints line'] = kpline
        json_data['k points'] = []
        
        for ik, KP in enumerate(self.kpoints):
            json_kpoint = KP.json()
            json_kpoint['kp in line'] = kpline[ik]
            if kpnames is None:
                json_kpoint['kpname'] = None
            else:
                json_kpoint['kpname'] = kpnames[ik]
            json_data['k points'].append(json_kpoint)
        
        json_data['indirect gap (eV)'] =  self.gap_indirect
        json_data['Minimal direct gap (eV)'] =  self.gap_direct

        if self.spinor and not self.magnetic:
            json_data["number of inversion-odd Kramers pairs"]  = int(self.num_bandinvs / 2)
            json_data["Z4"] = int(self.num_bandinvs / 2) % 4,
        else:
            json_data["number of inversion-odd states"]  = self.num_bandinvs

        try:
            json_data['symmetry indicators'] = self.symmetry_indicators
        except RuntimeError:  # irreps not identified beforehand
            pass

        if hasattr(self, 'classification'):  # compute_ebr_decomposition was run
            json_data['classification'] = self.classification
            json_data['ebr decomposition'] = {}
            json_data['ebr decomposition']['y'] = self.y
            json_data['ebr decomposition']['y_prime'] = self.y_prime
            json_data['ebr decomposition']['solutions'] = self.ebr_decompositions

        return json_data

    @property
    def gap_direct(self):
        '''
        Getter for the direct gap

        Returns
        -------
        gap : float
            Smallest direct gap
        '''

        gap = np.inf
        for KP in self.kpoints:
            gap = min(gap, KP.upper-KP.Energy_mean[-1])
        return gap

    @property
    def gap_indirect(self):
        '''
        Getter for the indirect gap

        Returns
        -------
        gap : float
            Smallest indirect gap
        '''

        min_upper = np.inf  # smallest energy of bands above set
        max_lower = -np.inf  # largest energy of bands in the set
        for KP in self.kpoints:
            min_upper = min(min_upper, KP.upper)
            max_lower = max(max_lower, KP.Energy_mean[-1])
        return min_upper - max_lower

    @property
    def num_bandinvs(self):
        '''
        Getter for the total number of inversion odd states

        Returns
        -------
        num_bandinvs : int
            Total number of inversion odd states. 0 if the crystal is not 
            inversion symmetric.
        '''

        num_bandinvs = 0
        for KP in self.kpoints:
            if KP.num_bandinvs is not None:
                num_bandinvs += KP.num_bandinvs
        return num_bandinvs

    def write_irrepsfile(self):
        '''
        Write the file `irreps.dat` with the identified irreps.
        '''

        file = open('irreps.dat', 'w')
        for KP in self.kpoints:
            KP.write_irrepsfile(file)
        file.close()


    @property
    def num_bands(self):
        """
        Return number of bands. Raise RuntimeError if the number of bands 
        varies from on k-point to the other.

        Returns
        -------
        int
            Number of bands in every k-point.
        """
        nbarray = [k.num_bands for k in self.kpoints]
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

    def write_trace(self,):
        """
        Generate `trace.txt` file to upload to the program `CheckTopologicalMat` 
        in `BCS <https://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl>`_ .
        """

        f = open("trace.txt", "w")
        f.write(
            (
                " {0}  \n"
                + " {1}  \n"  # Number of bands below the Fermi level  # Spin-orbit coupling. No: 0, Yes: 1
            ).format(self.num_bands, 1 if self.spinor else 0)
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
                    for x in KP.k
                )
                + "\n"
            )
        for KP in self.kpoints:
            f.write(
                KP.write_trace()
            )

    def Separate(self, isymop, groupKramers=True, verbosity=0):
        """
        Separate band structure according to the eigenvalues of a symmetry 
        operation.
        
        Parameters
        ----------
        isymop : int
            Index of symmetry used for the separation.
        groupKramers : bool, default=True
            If `True`, states will be coupled by Kramers' pairs.
        verbosity : int, default=0
            Verbosity level. Default is set to minimalistic printing

        Returns
        -------
        subspaces : dict
            Each key is an eigenvalue of the symmetry operation and the
            corresponding value is an instance of `class` `BandStructure` for 
            the subspace of that eigenvalue.
        """

        if isymop == 1:
            return {1: self}

        # Print description of symmetry used for separation
        symop = self.spacegroup.symmetries[isymop - 1]
        symop.show()

        # to do: allow for separation in terms of antiunitary symmetries
        if isymop > len(self.spacegroup.u_symmetries):
            raise RuntimeError("Separation in terms of antiunitary symmetries "
                               "not implemented for now.")

        # Separate each k-point
        kpseparated = [
            kp.Separate(symop, groupKramers=groupKramers, verbosity=verbosity, kwargs_kpoint=self.kwargs_kpoint)
            for kp in self.kpoints
        ] # each element is a dict with separated bandstructure of a k-point

        allvalues = np.array(sum((list(kps.keys()) for kps in kpseparated), []))
        if groupKramers:
            allvalues = allvalues[np.argsort(np.real(allvalues))].real
            block_indices = get_block_indices(allvalues, thresh=0.01, cyclic=False)
            if len(block_indices) > 1:
                allvalues = set(
                    [allvalues[b1:b2].mean() for b1, b2 in block_indices]
                ) # unrepeated Re parts of all eigenvalues
                subspaces = {}
                for vv in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(vv - vk))]
                        if abs(vk0 - vv) < 0.05:
                            other.kpoints.append(K[vk0])
                    subspaces[vv] = other # unnecessary indent ?
                return subspaces
            else:
                return dict({allvalues.mean(): self})
        else:
            allvalues = allvalues[np.argsort(np.angle(allvalues))]
            log_message(f'allvalues: {allvalues}', verbosity, 2)
            block_indices = get_block_indices(allvalues, thresh=0.01, cyclic=True)
            nv = len(allvalues)
            if len(block_indices) > 1:
                allvalues = set( [ np.roll(allvalues, -b1)[: (b2 - b1) % nv].mean()
                                    for b1, b2 in block_indices ])
                log_message(f'Distinct values: {allvalues}', verbosity, 2)
                subspaces = {}
                for vv in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(vv - vk))]
                        if abs(vk0 - vv) < 0.05:
                            other.kpoints.append(K[vk0])
                        subspaces[vv] = other
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
            (np.min([k.Energy[1:nmax] for k in self.kpoints], axis=0), [np.inf])
        )
        emax = np.max([k.Energy[:nmax] for k in self.kpoints], axis=0)
        locgap = np.hstack(
            (
                np.min(
                    [k.Energy[1:nmax] - k.Energy[0 : nmax - 1] for k in self.kpoints],
                    axis=0,
                ),
                [np.inf],
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
        wilson = functools.reduce(
            np.dot,
            [functools.reduce(np.dot, np.linalg.svd(O)[0:3:2]) for O in overlaps],
        )
        return np.sort((np.angle(np.linalg.eig(wilson)) / (2 * np.pi)) % 1)

    def write_plotfile(self, filename='bands-tognuplot.dat'):
        """
        Generate lines for a band structure plot, with cummulative length of the
        k-path as values for the x-axis and energy-levels for the y-axis.

        Returns
        -------
        str
            Lines to write into a file that will be parsed to plot the band 
            structure.
        """

        kpline = self.KPOINTSline()

        # Prepare energies at each k point
        energies_expanded = np.full((self.num_bands, len(kpline)), np.inf)
        for ik, kp in enumerate(self.kpoints):
            count = 0
            for iset, deg in enumerate(kp.degeneracies):
                for i in range(deg):
                    energies_expanded[count,ik] = kp.Energy_mean[iset]
                    count += 1

        # Write energies of each band
        file = open(filename, 'w')
        file.write('column 1: k, column 2: energy in eV (w.r.t. Fermi level)')
        for iband in range(self.num_bands):
            file.write('\n')  # blank line separating blocks of k points
            for k, energy in zip(kpline, energies_expanded[iband]):
                s = '{:8.4f}    {:8.4f}\n'.format(k, energy)
                file.write(s)
        file.close()

    def KPOINTSline(self, kpred=None, supercell=None, breakTHRESH=0.1):
        """
        Calculate cumulative length along a k-path in cartesian coordinates.
        ACCESSED DIRECTLY BY BANDUPPY>=0.3.4. DO NOT CHANGE UNLESS NECESSARY. 
        NOTIFY THE DEVELOPERS IF ANY CHANGE IS MADE.

        Parameters
        ----------
        kpred : list, default=None
            Each element contains the direct coordinates of a k-point in the
            attribute `kpoints`.
        supercell : array, shape=(3,3), default=None
                Describes how the lattice vectors of the (super)cell used in the 
                calculation are expressed in the basis vectors of the primitive 
                cell. USED IN BANDUPPY. DO NOT CHANGE UNLESS NECESSARY.
        breakTHRESH : float, default=0.1
            If the distance between two neighboring k-points in the path is 
            larger than `breakTHRESH`, it is taken to be 0. Set `breakTHRESH` 
            to a large value if the unforlded kpoints line is continuous.

        Returns
        -------
        K : array
            Each element is the cumulative distance along the path up to a 
            k-point. The first element is 0, so that the number of elements
            matches the number of k-points in the path.
        """
        if kpred is None:
            kpred = [k.k for k in self.kpoints]
        if supercell is None:
            reciprocal_lattice = self.RecLattice
        else:
            reciprocal_lattice = supercell.T @ self.RecLattice 
        KPcart = np.dot(kpred, reciprocal_lattice)
        K = np.zeros(KPcart.shape[0])
        k = np.linalg.norm(KPcart[1:, :] - KPcart[:-1, :], axis=1)
        k[k > breakTHRESH] = 0.0
        K[1:] = np.cumsum(k)
        return K


    def get_dmn(self, grid=None, degen_thresh=1e-2, unitary=True, unitary_params={}):
        """
        grid : tuple(int), optional
            the grid of kpoints (3 integers), if None, the grid is determined from the kpoints
            may be used to reduce the grid (by an integer factor) for the symmetry analysis
        degen_thresh : float, optional
            the threshold for the degeneracy of the bands. Only transformations between bands
             with energy difference smaller than this value are considered
        unitary : bool, optional
            if True, the transformation matrices are made unitary explicitly
        unitary_params : dict, optional
            parameters to be passed to :func:`~irrep.utility.orthogonalize`

        Returns
        -------
        dict with the following keys:
            grid : tuple(int)
                the grid of kpoints (3 integers - number of kpoints in each direction)
            kpoints : np.array((NK,3))
                the list of kpoints
            kptirr : list of int
                kptirr[i] is the index of the i-th irreducible kpoint in the list of kpoints
            kptirr2kpt : array of int
                kptirr2kpt[i,isym]=j means that the i-th irreducible kpoint
            kpt2kptirr : array of int
                kpt2kptirr[j]=i means that the j-th kpoint is the i-th irreducible kpoint
            d_band_blocks : list of list of list of array
                d_band_blocks[i][isym] is a list of transformation matrices 
                between (almost degenerate) bands at the i-th irreducible kpoint
                under the isym-th symmetry operation
            d_band_block_indices : list of list of int
                d_band_block_indices[i] is a list of indices of the bands that are almost degenerate
                at the i-th irreducible kpoint        
        """

        kpoints = np.array([KP.K  for KP in self.kpoints])
        grid, selected_kpoints = grid_from_kpoints(kpoints, grid=grid)
        kpoints = kpoints[selected_kpoints]
        Nsym = self.spacegroup.size

        def get_K(ik):
            return self.kpoints[selected_kpoints[ik]]

        # First determine which kpoints are irreducible
        # and set mapping from irreducible to full grid and back
        #
        # kptirr2kpt[i,isym]=j means that the i-th irreducible kpoint
        # is transformed to the j-th kpoint of full grid
        #  by the isym-th symmetry operation.
        #
        # This is consistent with w90 documentations, but seemd to be opposite to what pw2wannier90 does
        symmetries = self.spacegroup.symmetries

        kpoints_mod1 = UniqueListMod1(kpoints)
        assert len(kpoints_mod1) == len(kpoints)
        NK = len(kpoints_mod1)
        is_irreducible = np.ones(NK, dtype=bool)
        kptirr = []
        kptirr2kpt = []
        kpt2kptirr = -np.ones(NK, dtype=int)
        G = []
        ikirr = -1
        for i, k1 in enumerate(kpoints):
            if is_irreducible[i]:
                kptirr.append(i)
                kptirr2kpt.append(np.zeros(Nsym, dtype=int))
                G.append(np.zeros((Nsym, 3), dtype=int))
                ikirr += 1

                for isym, symop in enumerate(symmetries):
                    k1p = symop.transform_k(k1)
                    if k1p not in kpoints_mod1:
                        raise RuntimeError("Symmetry operation maps k-point outside the grid. Maybe the grid is incompatible with the symmetry operations")
                    j = kpoints_mod1.index(k1p)
                    k2 = kpoints[j]
                    if j != i:
                        is_irreducible[j] = False
                    kptirr2kpt[ikirr][isym] = j
                    # the G vectors mean that
                    # symop.transform(ki) = kj + G
                    G[ikirr][isym] = k1p - k2
                    if kpt2kptirr[j] == -1:
                        kpt2kptirr[j] = ikirr
                    else:
                        assert kpt2kptirr[j] == ikirr, (f"two different irreducible kpoints {ikirr} and {kpt2kptirr[j]} are mapped to the same kpoint {j}"
                                                             f"kptirr= {kptirr}, \nkpt2kptirr= {kpt2kptirr}\n kptirr2kpt= {kptirr2kpt}")
        kptirr = np.array(kptirr)
        NKirr = len(kptirr)
        kptirr2kpt = np.array(kptirr2kpt)
        G = np.array(G)
        del kpoints_mod1

        assert np.all(kptirr2kpt >= 0)
        assert np.all(kpt2kptirr >= 0)

        d_band_blocks = [[[] for _ in range(Nsym)] for _ in range(NKirr)]
        d_band_block_indices = []
        for i, ikirr in enumerate(kptirr):
            K1 = get_K(ikirr)
            block_indices = get_block_indices(K1.Energy_raw, thresh=degen_thresh, cyclic=False)
            d_band_block_indices.append(block_indices)
            for isym, symop in enumerate(symmetries):
                K2 = get_K(kptirr2kpt[i, isym])
                block_list = symm_matrix(
                    K=K1.k,
                    K_other=K2.k,
                    WF=K1.WF,
                    WF_other=K2.WF,
                    igall=K1.ig,
                    igall_other=K2.ig,
                    A=symop.rotation,
                    S=symop.spinor_rotation,
                    T=symop.translation,
                    time_reversal=symop.time_reversal,
                    spinor=K1.spinor,
                    block_ind=block_indices,
                    return_blocks=True,
                    unitary=unitary,
                    unitary_params=unitary_params,
                )
                d_band_blocks[i][isym] = [np.ascontiguousarray(b.T) for b in block_list]
                # transposed because in irrep WF is row vector, while in dmn it is column vector
        return dict(grid=grid, 
                    kpoints=kpoints,
                    kptirr=kptirr, 
                    kptirr2kpt=kptirr2kpt,
                    kpt2kptirr=kpt2kptirr, 
                    d_band_blocks=d_band_blocks, 
                    d_band_block_indices=d_band_block_indices)
        # return     (grid, 
        #             kpoints,
        #             kptirr, 
        #             kptirr2kpt,
        #             kpt2kptirr, 
        #             d_band_blocks, 
        #             d_band_block_indices)

    def get_irrep_counts(self, filter_valid=True):
        """
        Returns a dictionary with irrep labels as keys and multiplicity 
        as values.

        Parameters
        ----------
        filter_valid : bool, optional
            count only integer multiplicities, by default True
        """
        
        irrep_data = []
        for kpoint in self.kpoints:
            if 'None' in kpoint.irreps:
                raise RuntimeError(
                    "Could not get the irrep counts because irreps must be identified."
                )
            else:
                irrep_data.append(kpoint.irreps)

        # dictionary: {label of irrep: total multiplicity}
        irrep_dict = {}
        for point in irrep_data:
            for irrep in point:
                for label, multi in irrep.items():
                    # only add valid multiplicities (filter out uncoverged bands)
                    valid_multi = check_multiplicity(multi)
                    if valid_multi or (filter_valid is False):
                        multi = np.real(multi).round(0)
                        # If the irrep's label doesn't exist yet, create it
                        irrep_dict.setdefault(label, 0)
                        irrep_dict[label] += multi

        return irrep_dict

    @cached_property
    def symmetry_indicators(self):
        '''
        Lazy getter to return the symmetry indicators. 

        Returns
        -------
        indicators_dict : dict
            Keys and values are labels of indicators and their values, 
            respectively. If the space group doesn't have nontrivial 
            indicators, `None` is returned

        Raises
        ------
        RuntimeError
            If the irreps have not been identified beforehand via 
            `identify_irreps`
        '''

        # Identify_irreps must be used beforehand
        try:
            irrep_dict = self.get_irrep_counts()
        except RuntimeError:
            raise RuntimeError(
                "Could not compute the symmetry indicators "
                "because irreps must be identified. Try specifying -kpnames "
                "in the CLI (or run BandStructure.identify_irreps if you are "
                "using IrRep as a package"
            )

        # Load symmetry indicators file
        si_table = self.load_si_table()
        if self.spacegroup.number_str not in si_table:
            print("There are no non-trivial symmetry indicators for this space "
                  "group")
            indicators_dict = None

        else:
            si_table = si_table[self.spacegroup.number_str]["indicators"]

            indicators_dict = {}
            for indicator in si_table:
                si_factors = si_table[indicator]["factors"]
                total = 0
                for label, value in si_factors.items():
                    total += value * irrep_dict.get(label, 0)
                indicators_dict[indicator] = total % si_table[indicator]['mod']

        return indicators_dict

    def print_symmetry_indicators(self):
        """
        Computes and prints the symmetry-indicator information.

        Notes
        -----
        Method identify_irreps must have been called beforehand
        """

        print("\n---------- SYMMETRY INDICATORS ----------\n")

        if self.symmetry_indicators is None:
            print(f"Space group {self.spacegroup.name} has no nontrivial "
                    "indicators")

        else:
            si_table = self.load_si_table()
            si_table = si_table[self.spacegroup.number_str]["indicators"]

            for indicator in si_table:

                # String for the formula to calculate the indicator
                si_factors = si_table[indicator]["factors"]
                terms = [
                    f"{factor} x {label}" for label, factor in 
                    si_factors.items() if factor != 0
                ]
                definition_str = " + ".join(terms)
                
                print(f"{indicator} =", self.symmetry_indicators[indicator])
                print(f"\tDefinition: ({definition_str}) mod {si_table[indicator]['mod']}")

    def compute_ebr_decomposition(self):
        '''
        Compute EBR decomposition. Sets values for attributes `classification`, 
        `ebr_decompositions`, `y` and `y_prime`

        Raises
        ------
        RuntimeError
            Irreps were not identified beforehand by running `identify_irreps`
        '''

        from .ebrs import (
            compute_topological_classification_vector,
            ORTOOLS_AVAILABLE,
            compute_ebr_decomposition,
            load_ebr_data
        )

        # Load data from EBR files
        ebr_data = load_ebr_data(self.spacegroup.number_str, self.spinor)

        try:
            irrep_counts = self.get_irrep_counts()
        except RuntimeError:
            print(
                "Could not compute the EBR decomposition because counting of "
                "irreps failed."
            )

        (self.y,
         self.y_prime,
         nontrivial
        ) = compute_topological_classification_vector(irrep_counts, ebr_data)


        # Stable topological, don't compute EBR decompositions
        if nontrivial:
            self.classification = 'STABLE TOPOLOGICAL'
            self.ebr_decompositions = None
            return

        # Fragile or trivial, but cannot ortools not installed
        elif not ORTOOLS_AVAILABLE:
            self.ebr_decomposition = None
            print(
                "There exists integer-valued solutions to the EBR decomposition "
                "problem, so the set of bands is TRIVIAL or displays FRAGILE TOPOLOGY. "
                "Install OR-Tools to compute decompositions."
                )

        else:
            print('Calculating decomposition in terms of EBRs. '
                  'This can take some time...')
            self.ebr_decompositions, is_positive = compute_ebr_decomposition(ebr_data, self.y)
            if is_positive:
                self.classification = 'ATOMIC LIMIT'
            else:
                self.classification = 'FRAGILE TOPOLOGICAL'


    def print_ebr_decomposition(self):
        """
        Computes and prints the EBR decomposition information. If the bands are
        trivial or fragile-topological, it tries to find EBR decompositions using
        ORtools if installed.

        Notes
        -----
        The Smith decomposition follows this notation:

        .. math::

            EBR \cdot x = y, \\
            EBR = U^{-1} \cdot R \cdot R^{-1},\\
            R \cdot Y = C, \\
            x' = V^{-1} \cdot x,\\
            y' = U \cdot y.

        Raises
        ------
        RuntimeError
            Irreps were not identified beforehand by running `identify_irreps`
        """

        from .ebrs import (
            compose_irrep_string,
            get_ebr_names_and_positions,
            compose_ebr_string,
            get_smith_form,
            load_ebr_data
        )
        from .utility import vector_pprint

        # General block printed always
        print("\n---------- EBR DECOMPOSITION ----------\n")
        print(f'Classification: {self.classification}')

        try:
            irrep_counts = self.get_irrep_counts()
        except RuntimeError:
            print(
                "Could not compute the EBR decomposition because counting of "
                "irreps failed."
            )
        ebr_data = load_ebr_data(self.spacegroup.number_str, self.spinor)
        basis_labels = ebr_data["basis"]["irrep_labels"]
        _, d, _ = get_smith_form(ebr_data)
        smith_diagonal = d.diagonal()
        print(
        f"Irrep decomposition at high-symmetry points:\n\n{compose_irrep_string(irrep_counts)}"
        f"\n\nIrrep basis:\n{vector_pprint(basis_labels, fmt='s')}"
        f"\n\nSymmetry vector (y):\n{vector_pprint(self.y, fmt='d')}"
        f"\n\nTransformed symmetry vector (y'):\n{vector_pprint(self.y_prime, fmt='d')}"
        f"\n\nSmith singular values:\n{vector_pprint(smith_diagonal, fmt='d')}"
        f"\n\nNotation: EBR.x=y,  U.EBR.V=R,  y'=U.y"
        )

        # If EBR decomposition wasn't computed because ortools isn't installed
        if (self.classification != 'STABLE TOPOLOGICAL' 
            and self.ebr_decompositions is None):
            print(
                "There exists integer-valued solutions to the EBR decomposition "
                "problem, so the set of bands is TRIVIAL or displays FRAGILE TOPOLOGY. "
                "Install OR-Tools and compute decompositions again"
                )

        # If EBR decomposition was computed
        elif self.classification in ['ATOMIC LIMIT', 'FRAGILE TOPOLOGIVAL']:
            print('Printing EBR decompositions:')
            ebr_list = get_ebr_names_and_positions(ebr_data)
            for i, sol in enumerate(self.ebr_decompositions):
                print("Solution", i + 1, "\n")
                print(compose_ebr_string(sol, ebr_list), "\n")


    def load_si_table(self):
        '''
        Load table of symmetry indicators

        Returns
        -------
        dict
            Data loaded from the file of symmetry indicators
        '''

        root = os.path.dirname(__file__)
        filename = (
            f"{'double' if self.spinor else 'single'}_indicators"
            f"{'_magnetic' if self.magnetic else ''}.json"
            )
        si_table = json.load(open(root+"/data/symmetry_indicators/"+filename, 'r'))
        return si_table

        
def check_multiplicity(multi):
    """Checks if an irrep multiplicity is correct, i.e. integer.

    Parameters
    ----------
    multi : float
        irrep multiplicity

    Returns
    -------
    bool
        True if correct, else False
    """
    
    # is real
    if not np.isclose(np.imag(multi), 0, rtol=0, atol=1e-3):
        return False
    # is integer
    if not np.isclose(multi, np.round(multi), rtol=0, atol=1e-3):
        return False

    return True
