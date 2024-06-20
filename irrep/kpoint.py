
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


import numpy as np
import numpy.linalg as la
import copy
from .gvectors import symm_eigenvalues, NotSymmetryError, symm_matrix
from .utility import compstr, is_round, format_matrix
from lazy_property import LazyProperty

class Kpoint:
    """
    Parses files and organizes info about the states and energy-levels of a 
    particular k-point in attributes. Contains methods to calculate and write 
    traces (and irreps), for the separation of the band structure in terms of a 
    symmetry operation and for the calculation of the Zak phase.

        symmetries=None,
        symmetries_tables=None  # calculate_traces needs it
    Parameters
    ----------
    ik : int
        Index of kpoint, starting count from 0.
    NBin : int
        Number of bands considered at the k-point in the DFT calculation.
    IBstart : int
        First band to be considered.
    IBend : int
        Last band to be considered.
    Ecut : float
        Plane-wave cutoff (in eV) to consider in the expansion of wave-functions.
    Ecut0 : float
        Plane-wave cutoff (in eV) used in the DFT calulation. Always read from 
        DFT files. Insignificant if `code`='wannier90'.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    symmetries_SG : list, default=None
        Each element is an instance of class `SymmetryOperation` corresponding 
        to a symmetry in the point group of the space-group. The value 
        passed is the attribute `symmetries` of class `SpaceGroup`.
    spinor : bool, default=None
        `True` if wave functions are spinors, `False` if they are scalars.
    kpt : list or array, default=None
        Direct coordinates of the k-point.
    Energy : array
        Energy levels of the states with band indices between `IBstart` and 
        `IBend`.
    ig : array
        Array returned by :func:`~gvectors.sortIG`. It contains data about the 
        plane waves in the expansion of wave functions.
    upper : float
        Energy of the state `IBend`+1. Used to calculate the gap with upper 
        bands.
    degen_thresh : float, default=1e-8
        Threshold to identify degenerate energy levels.
    refUC : array, default=None
        3x3 array describing the transformation of vectors defining the 
        unit cell to the standard setting.
    shiftUC : array, default=None
        Translation taking the origin of the unit cell used in the DFT 
        calculation to that of the standard setting.
    symmetries_tables : list
        Attribute `symmetries` of class `IrrepTable`. Each component is an 
        instance of class `SymopTable` corresponding to a symmetry operation
        in the "point-group" of the space-group.
    
    Attributes
    ----------
    spinor : bool
        `True` if wave-functions are spinors, `False` if they are scalars.
    ik0 : int
        Index of the k-point, starting the count from 1.
    Nband : int
        Number of bands whose traces should be calculated.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    WF : array
        Coefficients of wave-functions in the plane-wave expansion. A row for 
        each wave-function, a column for each plane-wave.
    ig : array
        Returned by `sortIG`.
        Every column corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of rows is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth row stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) row contains the index of the first (last) plane-wave with 
        the same energy as the plane-wave of the current column.
    K : array, shape=(3,)
        Direct coordinates of the k point in the DFT cell setting.
    k_refUC : array, shape=(3,)
        Direct coordinates of the k point in the reference cell setting.
    Energy : array
        Energy-levels of bands whose traces should be calculated.
    upper : float
        Energy of the first band above the set of bands whose traces should be 
        calculated. It will be set to `numpy.NaN` if the last band matches the 
        last band in the DFT calculation (default `IBend`).
    symmetries_SG : list
        Each element is an instance of class `SymmetryOperation` corresponding 
        to a symmetry in the point group of the space-group. The value 
        passed is the attribute `symmetries` of class `SpaceGroup`.
    symmetries : dict
        Each key is an instance of `class` `SymmetryOperation` corresponding 
        to an operation in the little-(co)group and the attached value is an 
        array with the traces of the operation.
    char : array
        Each row corresponds to a set of degenerate states. Each column is the 
        trace of a symmetry in the little cogroup in the DFT cell setting.
    char_refUC : array
        The same as `char`, but in the reference cell setting.
    degeneracies : array
        Degeneracies of energy levels between `IBstart` and `IBend`.
    borders : array
        Integers representing the band index of the first state in each set of 
        degenerate states. The bounds can be obtained as 
        `for ibot, itop in zip(borders[:-1], borders[1:])`.
    Energy_mean : array
        Average of energy levels within each set of degenerate states
    num_bandinvs : int
        Property getter for the number of inversion-odd states. If the 
        k point is not inversion symmetric, `None` is returned.
    NG : int
        Property getter for the number of plane waves in the expansion of 
        wave functions.
    onlytraces : bool
        `False` if irreps have been identified and have to be written.
    """

    def __init__(
        self,
        ik=None,
        IBstart=None,
        IBend=None,
        RecLattice=None,  # this was last mandatory argument
        symmetries_SG=None,
        spinor=None,
        kpt=None,
        WF=None,  # first arg added for abinit (to be kept at the end)
        Energy=None,
        ig=None,
        upper=None,
        degen_thresh=1e-8,
        symmetries=None,
        refUC=np.eye(3),
        shiftUC=np.zeros(3),
        symmetries_tables=None  # calculate_traces needs it
    ):
        self.spinor = spinor
        self.ik0 = ik + 1  # the index in the WAVECAR (count start from 1)
        self.Nband = IBend - IBstart
        #        self.n=np.arange(IBstart,IBend)+1
        self.RecLattice = RecLattice
        self.symmetries_SG = symmetries_SG  # lazy_property needs it
        self.upper = upper

        self.K = kpt
        self.WF = WF
        self.Energy = Energy
        self.ig = ig
        self.upper = upper

        self.k_refUC = np.dot(refUC.T, self.K)
        self.WF /= (
            np.sqrt(np.abs(np.einsum("ij,ij->i", self.WF.conj(), self.WF)))
        ).reshape(self.Nband, 1)

        # Calculate traces
        self.char, self.char_refUC, self.Energy, self.degeneracies, self.borders = self.calculate_traces(refUC, shiftUC, symmetries, symmetries_tables, degen_thresh)

        # Determine number of band inversions based on parity
        found = False
        for sym in self.symmetries:
            if (
                sum(abs(sym.translation)) < 1e-6
                and
                abs(sym.rotation + np.eye(3)).sum() < 1e-6
            ):
                found = True
                break
        if found:
            self.num_bandinvs = int(round(sum(1 - self.symmetries[sym].real) / 2))
        else:
            self.num_bandinvs = None

    # creates attribute symmetries, if it was not created before
    @LazyProperty
    def symmetries(self):
        """
        Sets the attribute `Kpoint.symmetries` to a dictionary. Works as a 
        lazy-property.

        Returns
        -------
        symmetries : dict
            Each key is an instance of `class` `SymmetryOperation` corresponding 
            to an operation in the little-(co)group and the attached value is an 
            array with the traces of the operation.
    
        Notes
        -----
        For more about `lazy-property`, check the documentation `here <https://pypi.org/project/lazy-property/>`_ .
        """
        symmetries = {}
        #        print ("calculating symmetry eigenvalues for E={0}, WF={1} SG={2}".format(self.Energy,self.WF.shape,symmetries_SG) )
        if not (self.symmetries_SG is None):
            for symop in self.symmetries_SG:
                try:
                    symmetries[symop] = symm_eigenvalues(
                        self.K,
                        self.RecLattice,
                        self.WF,
                        self.ig,
                        symop.rotation,
                        symop.spinor_rotation,
                        symop.translation,
                        self.spinor,
                    )
                except NotSymmetryError:
                    pass
        return symmetries

    def copy_sub(self, E, WF, inds):
        """
        Create an instance of class `Kpoint` for a restricted set of states.

        Parameters
        ----------
        E : array
            Energy-levels of states.
        WF : array
            Coefficients of the plane-wave expansion of wave-functions. Each row 
            corresponds to a wave-function, each row to a plane-wave.

        Returns
        -------
        other : class
            Instance of `Kpoints` corresponding to the group of states passed. 
            They are shorted by energy-levels.
        """

        other = copy.copy(self) # copy of whose class

        # Sort energy levels
        sortE = np.argsort(E)
        other.Energy = E[sortE]
        other.WF = WF[sortE]
        other.Nband = len(E)
        inds = inds[sortE]

        # Do not group by degeneracy of energy-levels for printing
        other.degeneracies = [1] * self.Nband
        char = []
        char_refUC = []
        for i in inds:
            char.append(other.char[i])
            char_refUC.append(other.char_refUC[i])
        other.char = np.array(char)
        other.char_refUC = np.array(char_refUC)
        irreps = []
        for i in inds:
            irreps.append(other.irreps[i])
        other.irreps = irreps
        return other

    def unfold(self, supercell, kptPBZ, degen_thresh=1e-4):
        """
        Unfolds a kpoint of a supercell onto the point of the primitive cell 
        `kptPBZ`.

        Parameters
        ----------
        supercell : array, shape=(3,3)
            Describes how the lattice vectors of the (super)cell used in the 
            calculation are expressed in the basis vectors of the primitive 
            cell.
        kptPBZ : array, shape=(3,)
            Coordinates of the k-point in the primitive Brillouin zone (PBZ), 
            on which the present kpoint should be unfolded.
        degen_thresh : float
            Bands with energy difference smaller that the threshold will be 
            considered as one band, and only one total weight will be given for 
            them.

        Returns
        -------
        array
            Array containing 2 columns (5 for spinor case): E, W, Sx, Sy, Sz.
            E - energy of the band or average energy of the group of bands.
            W - weight of the band(s) projected onto the PBZ kpoint.
            Sx, Sy, Sz - Spin components projected onto the PBZ kpoint.
        """
        if not is_round(kptPBZ.dot(supercell.T) - self.K, prec=1e-5):
            raise RuntimeError(
                "unable to unfold {} to {}, withsupercell={}".format(
                    self.K, kptPBZ, supercell
                )
            )
        g_shift = kptPBZ - self.K.dot(np.linalg.inv(supercell.T))
        #        print ("g_shift={}".format(g_shift))
        selectG = np.array(
            np.where(
                [
                    is_round(dg, prec=1e-4)
                    for dg in (self.ig[:3].T.dot(np.linalg.inv(supercell.T)) - g_shift)
                ]
            )[0]
        )
        #        print ("unfolding {} to {}, selecting {} of {} g-vectors \n".format(self.K,kptPBZ,len(selectG),self.ig.shape[1],selectG,self.ig.T))
        if self.spinor:
            selectG = np.hstack((selectG, selectG + self.NG))
        WF = self.WF[:, selectG]
        result = []
        for b1, b2, E, matrices in self.get_rho_spin(degen_thresh):
            proj = np.array(
                [
                    [WF[i].conj().dot(WF[j]) for j in range(b1, b2)]
                    for i in range(b1, b2)
                ]
            )
            result.append([E,] + [np.trace(proj.dot(M)).real for M in matrices])
        return np.array(result)

    def get_rho_spin(self, degen_thresh=1e-4):
        """ 
        A getter, made to avoid the repeated evaluation of 
        self.__eval_rho_spin for the same degen_thresh.

        Parameters
        ----------
        degen_thresh : float
            Bands with energy difference smaller that the threshold will be 
            considered as one band, and only one total weight will be given for 
            them.
        """
        if not hasattr(self, "rho_spin"):
            self.rho_spin = {}
        if degen_thresh not in self.rho_spin:
            self.rho_spin[degen_thresh] = self.__eval_rho_spin(degen_thresh)
        return self.rho_spin[degen_thresh]

    @property
    def NG(self):
        """Getter for the number of plane-waves in current k-point"""
        return self.ig.shape[1]

    def __eval_rho_spin(self, degen_thresh):
        """
        Evaluates the matrix <i|M|j> in every group of degenerate 
        bands labeled by i and j, where M is :math:`\sigma_0`, 
        :math:`\sigma_x`, :math:`\sigma_y` or :math:`\sigma_z` for 
        the spinor case, :math:`\sigma_0` for the spinor case.

        Parameters
        ----------
        degen_thresh : float
            Bands with energy difference smaller that the threshold will 
            be considered as one group.

        Returns
        -------
        list of tuples
            Each tuple contains  b1 ,b2, E, (M , Sx , Sy , Sz), where
            b1 - index of first band in the group
            b2 - index of the last band inthe group +1
            E - average energy of the group 
            M -  <i|j>
            Sx, Sy, Sz - <i|sigma|j> 
        """
        borders = np.hstack(
            [
                [0],
                np.where(self.Energy[1:] - self.Energy[:-1] > degen_thresh)[0] + 1,
                [self.Nband],
            ]
        )
        result = []
        for b1, b2 in zip(borders, borders[1:]):
            E = self.Energy[b1:b2].mean()
            W = np.array(
                [
                    [self.WF[i].conj().dot(self.WF[j]) for j in range(b1, b2)]
                    for i in range(b1, b2)
                ]
            )
            if self.spinor:
                ng = self.NG
                Smatrix = [
                    [
                        np.array(
                            [
                                [
                                    self.WF[i, ng * s : ng * (s + 1)]
                                    .conj()
                                    .dot(self.WF[j, ng * t : ng * (t + 1)])
                                    for j in range(b1, b2)
                                ]
                                for i in range(b1, b2)
                            ]
                        )  # band indices
                        for t in (0, 1)
                    ]
                    for s in (0, 1)
                ]  # spin indices
                Sx = Smatrix[0][1] + Smatrix[1][0]
                Sy = 1j * (-Smatrix[0][1] + Smatrix[1][0])
                Sz = Smatrix[0][0] - Smatrix[1][1]
                result.append((b1, b2, E, (W, Sx, Sy, Sz)))
            else:
                result.append((b1, b2, E, (W,)))
        return result

    def Separate(self, symop, degen_thresh, groupKramers=True):
        """
        Separate the band structure in a particular k-point according to the 
        eigenvalues of a symmetry operation.

        Parameters
        ----------
        isymop : int
            Index of symmetry used for the separation.
        degen_thresh : float, default=1e-5
            Energy threshold used to determine degeneracy of energy-levels.
        groupKramers : bool, default=True
            If `True`, states will be coupled by pairs of Kramers.

        Returns
        -------
        subspaces : dict
            Each key is an eigenvalue of the symmetry operation and the
            corresponding value is an instance of `class` `Kpoint` for the 
            states with that eigenvalue.
        """

        # Check orthogonality of wave functions
        # Rm once tests are fixed
        norms = self.WF.conj().dot(self.WF.T)
        check = np.max(abs(norms - np.eye(norms.shape[0])))
        if check > 1e-5:
            print(
                "orthogonality (largest of diag. <psi_nk|psi_mk>): {0:7.5} > 1e-5   \n".format(
                    check
                )
            )

        S = symm_matrix(
            self.K,
            self.RecLattice,
            self.WF,
            self.ig,
            symop.rotation,
            symop.spinor_rotation,
            symop.translation,
            self.spinor,
        )


        # Check that S is block-diagonal
        Sblock = np.copy(S)
        for b1, b2 in zip(self.borders, self.borders[1:]):
            Sblock[b1:b2, b1:b2] = 0
        check = np.max(abs(Sblock))
        if check > 0.1:
            print(("WARNING: matrix of symmetry has non-zero elements between "
                   "states of different energy:  \n", check))
            print("Printing matrix of symmetry at k={}".format(self.K))
            print(format_matrix(Sblock))

        # Calculate eigenvalues and eigenvectors in each block
        eigenvalues = []
        eigenvectors = []
        inds_states = []
        Eloc = []
        for istate, num_states in  enumerate(self.degeneracies):
            b1 = self.borders[istate]
            b2 = self.borders[istate+1]
            inds_states += [istate] * num_states  # index for set of states
            W, V = la.eig(S[b1:b2, b1:b2])
            for w, v in zip(W, V.T):
                eigenvalues.append(w)
                Eloc.append(self.Energy[istate])
                eigenvectors.append(
                    np.hstack((np.zeros(b1), v, np.zeros(self.Nband - b2)))
                )
        w = np.array(eigenvalues)
        v = np.array(eigenvectors).T # each col an eigenvector
        Eloc = np.array(Eloc)
        inds_states = np.array(inds_states)

        # Check unitarity of the symmetry
        if np.abs((np.abs(w) - 1.0)).max() > 1e-4:
            print("WARNING : some eigenvalues are not unitary :{0} ".format(w))
        if np.abs((np.abs(w) - 1.0)).max() > 3e-1:
            raise RuntimeError(" some eigenvalues are not unitary :{0} ".format(w))
        w /= np.abs(w)

        subspaces = {}

        if groupKramers:

            # Sort based on real part of eigenvalues
            arg = np.argsort(np.real(w))
            w = w[arg]
            v = v[:, arg]
            Eloc = Eloc[arg]
            inds_states = inds_states[arg]
            borders = np.hstack(
                ([0], np.where((w[1:] - w[:-1]) > 0.05)[0] + 1, [self.Nband])
            )

            # Probably this if-else statement can be removed
            if len(borders) > 0:
                for b1, b2 in zip(borders, borders[1:]):
                    v1 = v[:, b1:b2]
                    print(w[b1:b2].mean())
                    subspaces[w[b1:b2].mean()] = self.copy_sub(E=Eloc[b1:b2], WF=v1.T.dot(self.WF), inds=inds_states[b1:b2])
            else:
                v1 = v
                subspaces[w.mean()] = self.copy_sub(E=Eloc, WF=v1.T.dot(self.WF), degen_thresh=degen_thresh, inds_states=inds_states)

        else:  # don't group Kramers pairs
            
            # Sort based on the argument of eigenvalues
            arg = np.argsort(np.angle(w))
            w = w[arg]
            v = v[:, arg]
            Eloc = Eloc[arg]
            borders = np.where(abs(w - np.roll(w, 1)) > 0.1)[0]

            if len(borders) > 0:
                for b1, b2 in zip(borders, np.roll(borders, -1)):
                    v1 = np.roll(v, -b1, axis=1)[:, : (b2 - b1) % self.Nband]
                    subspaces[np.roll(w, -b1)[: (b2 - b1) % self.Nband].mean()] = self.copy_sub(
                        E=np.roll(Eloc, -b1)[: (b2 - b1) % self.Nband], degen_thresh=degen_thresh, WF=v1.T.dot(self.WF)
                    )

            else:
                v1 = v
                subspaces[w.mean()] = self.copy_sub(E=Eloc, degen_thresh=degen_thresh, WF=v1.T.dot(self.WF))

        return subspaces

    def calculate_traces(self, refUC, shiftUC, symmetries, symmetries_tables, degen_thresh=1e-8):
        '''
        Calculate traces of symmetry operations

        Parameters
        ----------
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=None
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        symmetries : list
            Indices of symmetries whose traces will be calculated.
        symmetries_tables : list
            Attribute `symmetries` of class `IrrepTable`. Each component is an 
            instance of class `SymopTable` corresponding to a symmetry operation
            in the "point-group" of the space-group.
        degen_thresh : float, default=1e-8
            Threshold to identify degenerate energy levels.
        
        Returns
        -------
        char : array
            Each row corresponds to a set of degenerate states. Each column is the 
            trace of a symmetry in the little cogroup in the DFT cell setting.
        char_refUC : array
            The same as `char`, but in the reference cell setting.
        Energy_mean : array
            Average of energy levels within each set of degenerate states
        degeneracies : array
            Degeneracies of energy levels between `IBstart` and `IBend`.
        borders : array
            Integers representing the band index of the first state in each set of 
            degenerate states. The bounds can be obtained as 
            `for ibot, itop in zip(borders[:-1], borders[1:])`.
        '''

        if symmetries is None:
            sym = {s.ind: s for s in self.symmetries}
        else:
            sym = {s.ind: s for s in self.symmetries if s.ind in symmetries}

        # Put all traces in an array. Rows (cols) correspond to syms (wavefunc)
        char = np.vstack([self.symmetries[sym[i]] for i in sorted(sym)])
        borders = np.hstack(
            [
                [0],
                np.where(self.Energy[1:] - self.Energy[:-1] > degen_thresh)[0] + 1,
                [self.Nband],
            ]
        )
        degeneracies = borders[1:] - borders[:-1]

        # Check that number of irreps is int
        Nirrep = np.linalg.norm(char.sum(axis=1)) ** 2 / char.shape[0]
        if abs(Nirrep - round(Nirrep)) > 1e-2:
            print("WARNING - non-integer number of states : {0}".format(Nirrep))
        Nirrep = int(round(Nirrep))

        # Sum traces of degenerate states. Rows (cols) correspond to states (syms)
        char = np.array(
            [char[:, start:end].sum(axis=1) for start, end in zip(borders, borders[1:])]
            )

        # Take average of energies over degenerate states
        Energy_mean = np.array(
            [self.Energy[start:end].mean() for start, end in zip(borders, borders[1:])]
        )

        # Transfer traces in calculational cell to refUC
        char_refUC = char.copy()
        if (not np.allclose(refUC, np.eye(3, dtype=float)) or
            not np.allclose(shiftUC, np.zeros(3, dtype=float))):
            # Calculational and reference cells are not identical
            for i,ind in enumerate(sym):
                dt = (symmetries_tables[ind-1].t 
                      - sym[ind].translation_refUC(refUC, shiftUC))
                char_refUC[:,i] *= (sym[ind].sign 
                                     * np.exp(-2j*np.pi*dt.dot(self.k_refUC)))

        return char, char_refUC, Energy_mean, degeneracies, borders


    def identify_irreps(self, irreptable=None):
        '''
        Identify irreps based on traces. Sets attributes `onlytraces` and  
        `irreps`.

        Parameters
        ----------
        irreptable : dict
            Each key is the label of an irrep, each value another `dict`. Keys 
            of every secondary `dict` are indices of symmetries (starting from 
            1 and following order of operations in tables of BCS) and 
            values are traces of symmetries.
        '''

        self.onlytraces = irreptable is None
        if self.onlytraces:
            irreps = ["None"] * (len(self.degeneracies) - 1)

        else:

            # irreps is a list. Each element is a dict corresponding to a 
            # group of degen. states. Every key is an irrep and its value 
            # the multiplicity of the irrep in the rep. of degen. states
            try:
                irreps = []
                for ch in self.char:
                    multiplicities = {}
                    for ir in irreptable:
                        multipl = np.dot(np.array([irreptable[ir][sym.ind] for sym in self.symmetries]),
                                         ch.conj()
                                         ) / len(ch)
                        if abs(multipl) > 1e-3:
                            multiplicities[ir] = multipl
                    irreps.append(multiplicities)
            except KeyError as ke:
                print(ke)
                print("irreptable:", irreptable)
                print([sym.ind for sym in self.symmetries])
                raise ke

        self.irreps = irreps


    def write_characters(self):
        '''
        Write the block of data of the k point, including energy levels,
        degeneracies, traces and irreps.
        '''

        # Print header for k-point
        print(("\n\n k-point {0:3d} : {1} (in DFT cell)\n"
               "               {2} (after cell trasformation)\n\n"
               " number of states : {3}\n"
               .format(self.ik0,
                       np.round(self.K, 5),
                       np.round(self.k_refUC, 5),
                       self.Nband)
              ))

        # Generate str describing irrep corresponding to sets of states
        str_irreps = []
        for irreps in self.irreps:  # set of IRs for a set of degenerate states
            if self.onlytraces:
                s = '  None  '
            else:
                s = ''
                for ir in irreps:  # label and multiplicity of one irrep
                    if s != '':
                        s += ', '  # separation between labels for irreps
                    s += ir
                    s += '({0:.5}'.format(irreps[ir].real)
                    if abs(irreps[ir].imag) > 1e-4:
                        s += '{0:+.5f}i'.format(irreps[ir].imag)
                    s += ')'
            str_irreps.append(s)

        # Set auxiliary blank strings for formatting
        writeimaginary = np.abs(self.char.imag).max() > 1e-4
        if writeimaginary:
            aux1 = ' ' * 4
        else:
            aux1 = ''
        irreplen = max(len(irr) for irr in str_irreps)
        #if irreplen % 2 == 1:
        #    irreplen += 1
        #aux2 = " " * int(irreplen / 2 - 3)
        num_spaces = (irreplen-8) / 2
        aux2 = " " * int(num_spaces)
        if irreplen % 2 == 0:
            aux3 = aux2 
        else:
            aux3 = aux2 + " "

        print("   Energy  |   degeneracy  | {0} irreps {1} | sym. operations  ".format(aux2, aux3))

        # Print indices of little-group symmetries
        s = "           |               | {0}        {1} | ".format(aux2, aux3)
        inds = []
        for sym in self.symmetries:
            inds.append(aux1 + "{0:4d}    ".format(sym.ind) + aux1)
        s += " ".join(inds)
        print(s)

        # Print line associated to a set of degenerate states
        for e, d, ir, ch1, ch2 in zip(self.Energy, self.degeneracies, str_irreps, self.char, self.char_refUC):

            # Traces in DFT unit cell
            right_str1 = []
            right_str2 = []
            for tr1, tr2 in zip(ch1, ch2):
                s1 = "{0:8.4f}".format(tr1.real)
                s2 = "{0:8.4f}".format(tr2.real)
                if writeimaginary:
                    s1 += "{0:+7.4f}j".format(tr1.imag)
                    s2 += "{0:+7.4f}j".format(tr2.imag)
                right_str1.append(s1)
                right_str2.append(s2)
            right_str1 = ' '.join(right_str1)
            right_str2 = ' '.join(right_str2)

            # Energy, degeneracy, irrep's label and character in DFT cell
            left_str = (" {0:8.4f}  |    {1:5d}      | {2:{3}s} |"
                        .format(e, d, ir, irreplen)
                        )
            print(left_str + " " + right_str1)

            # Line for character in reference cell
            left_str = ("           |               | {0:{1}s} |"
                        .format(len(ir)*" ", irreplen)
                        )
            print(left_str + " " + right_str2)  # line for character in DFT


    def json(self, symmetries=None):
        '''
        Prepare the data to save it in JSON format.

        Parameters
        ----------
        symmetries : list
            Indices of symmetries whose traces should be written.

        Returns
        -------
        json_data : dict
            Data that will be saved in JSON format.
        '''

        json_data = {}

        # Dictionary with symmetry eigenvalues (do we need this?)
        if symmetries is None:
            sym = {s.ind: s for s in self.symmetries}
        else:
            sym = {s.ind: s for s in self.symmetries if s.ind in symmetries}
        json_data ['symmetries'] = list(sym.keys())

        # Energy levels and degeneracies
        json_data['energies'] = self.Energy
        json_data['dimensions'] = self.degeneracies

        # Irreps and multiplicities
        if self.onlytraces:
            json_data['irreps'] = None 
        else:
            json_data['irreps'] = []
            for state in self.irreps:
                d = {}
                for irrep, multipl in state.items():
                    d[irrep] = (multipl.real, multipl.imag)
                json_data['irreps'].append(d)

        # Traces of symmetries
        json_data['characters'] = self.char
        json_data['characters_refUC'] = self.char_refUC
        if np.allclose(self.char, self.char_refUC, rtol=0.0, atol=1e-4):
            json_data['characters_refUC_is_the_same'] = True
        else:
            json_data['characters_refUC_is_the_same'] = False
        
        return json_data
        
    def write_irrepsfile(self, file):
        '''
        Write the irreps of this k point into `irreps.dat` file.

        Parameters
        ----------
        file : File object
            File object for the `irreps.dat` file.
        '''

        for energy, irrep_dict in zip(self.Energy, self.irreps):
            irrep = ''.join(irrep_dict.keys())
            s = '{:15.7f}    {:15s}\n'.format(energy, irrep)
            file.write(s)


    def write_plotfile(self, kpl, efermi):

        writeimaginary = np.abs(self.character.imag).max() > 1e-4
        s = []
        for e, dim, char in zip(self.Energy, self.degeneracies, self.character):
            s_loc = '{2:8.4f}   {0:8.4f}      {1:5d}   '.format(e-efermi, dim, kpl)
            for tr in char:
                s_loc += "{0:8.4f}".format(tr.real)
                if writeimaginary:
                    s_loc += "{0:+7.4f}j ".format(tr.imag)
            s.append(s_loc)
        s = '\n'.join(s)
        s += '\n\n'


    def write_irrepfile(self, firrep):

        file = open(firrep, "a")
        for e, ir in zip(self.Energy, self.irreps):
            for irrep in ir.split(","):
                try:
                    weight = abs(compstr(irrep.split("(")[1].strip(")")))
                    if weight > 0.3:
                        file.write(
                            " {0:10s} ".format(irrep.split("(")[0])
                            + "  {0:10.5f}\n".format(e)
                        )
                except IndexError:
                    pass
        file.close()



    def write_trace(self):
        """
        Write in `trace.txt` the block corresponding to a single k-point.

        Parameters
        ----------
        degen_thresh : float, default=1e-8
            Threshold energy used to decide whether wave-functions are
            degenerate in energy.
        symmetries : list, default=None
            Index of symmetry operations whose traces will be printed. 
        efermi : float, default=0.0
            Fermi-energy. Used as origin for energy-levels. 

        Returns
        -------
        res : str
            Block to write in `trace.txt` with description of traces in a
            single k-point.
        """

        sym = {s.ind: s for s in self.symmetries}
        res = (
            "{0} \n"
            + " {1} \n"  # Number of symmetry operations of the little co-group of the 1st maximal k-vec. In the next line the position of each element of the point group in the list above.
            # For each band introduce a row with the followind data: (1) 1+number of bands below, (2) dimension (degeneracy) of the band,
            # (3) energy and eigenvalues (real part, imaginary part) for each symmetry operation of the little group (listed above).
        ).format(len(sym.keys()), "  ".join(str(x) for x in sym))

        IB = np.cumsum(np.hstack(([0], self.degeneracies[:-1]))) + 1
        res += (
            "\n".join(
                (" {ib:8d}  {d:8d}   {E:8.4f} ").format(E=e, d=d, ib=ib)
                + "  ".join("{0:10.6f}   {1:10.6f} ".format(c.real, c.imag) for c in ch)
                for e, d, ib, ch in zip(self.Energy, self.degeneracies, IB, self.char)
            )
            + "\n"
        )

        return res

    def write_trace_all(self, degen_thresh=1e-8, symmetries=None, efermi=0.0, kpline=0):
        """
        Generate a block describing energy-levels and traces in a k-point.

        Parameters
        ----------
        degen_thresh : float, default=1e-8
            Threshold energy used to decide whether wave-functions are
            degenerate in energy.
        symmetries : list, default=None
            Index of symmetry operations whose traces will be printed. 
        efermi : float, default=0.0
            Fermi-energy. Used as origin for energy-levels. 
        kpline : float, default=0
            Cumulative length of the path up to current k-point.

        Returns
        -------
        str
            Block with the description of energy-levels and traces in a k-point.
        """
        preline = "{0:10.6f}     {1:10.6f}  {2:10.6f}  {3:10.6f}  ".format(
            kpline, *tuple(self.K)
        )
        if symmetries is None:
            sym = {s.ind: s for s in self.symmetries}
        else:
            sym = {s.ind: s for s in self.symmetries if s.ind in symmetries}

        char0 = {i: self.symmetries[sym[i]] for i in sym}
        borders = np.hstack(
            [
                [0],
                np.where(self.Energy[1:] - self.Energy[:-1] > degen_thresh)[0] + 1,
                [self.Nband],
            ]
        )
        char = {
            i: np.array(
                [char0[i][start:end].sum() for start, end in zip(borders, borders[1:])]
            )
            for i in char0
        } # keys are indices of symmetries, values are arrays with traces
        E = np.array(
            [self.Energy[start:end].mean() for start, end in zip(borders, borders[1:])]
        )
        dim = np.array([end - start for start, end in zip(borders, borders[1:])])
        IB = np.cumsum(np.hstack(([0], dim[:-1]))) + 1
        res = (
            "\n".join(
                preline
                + (" {ib:8d}  {d:8d}   {E:8.4f} ").format(E=e - efermi, d=d, ib=ib)
                + "     ".join(
                    (
                        "{0:10.6f} {1:10.6f}".format(char[i][j].real, char[i][j].imag)
                        if i in char
                        else (" " * 7 + "X" * 3 + " " * 8 + "X" * 3)
                    )
                    for i in range(1, len(self.symmetries_SG) + 1)
                )
                for e, d, ib, j in zip(E, dim, IB, np.arange(len(dim)))
            )
            + "\n"
        )
        return res

    def overlap(self, other):
        """ 
        Calculates the overlap matrix of elements < u_m(k) | u_n(k+g) >.

        Parameters
        ----------
        other : class
            Instance of `Kpoints` corresponding to `k+g` (next k-point in path).

        Returns
        -------
        res : array
            Matrix of `complex` elements  < u_m(k) | u_n(k+g) >.
        """
        g = np.array((self.K - other.K).round(), dtype=int)
        igall = np.hstack((self.ig[:3], other.ig[:3] - g[:, None]))
        igmax = igall.max(axis=1)
        igmin = igall.min(axis=1)
        igsize = igmax - igmin + 1
        #        print (self.ig.T)
        #        print (igsize)
        res = np.zeros((self.Nband, other.Nband), dtype=complex)
        
        # short again coefficients of expansions
        for s in [0, 1] if self.spinor else [0]:
            WF1 = np.zeros((self.Nband, igsize[0], igsize[1], igsize[2]), dtype=complex)
            WF2 = np.zeros(
                (other.Nband, igsize[0], igsize[1], igsize[2]), dtype=complex
            )
            for i, ig in enumerate(self.ig.T):
                WF1[:, ig[0] - igmin[0], ig[1] - igmin[1], ig[2] - igmin[2]] = self.WF[
                    :, i + s * self.ig.shape[1]
                ]
            for i, ig in enumerate(other.ig[:3].T - g[None, :]):
                WF2[:, ig[0] - igmin[0], ig[1] - igmin[1], ig[2] - igmin[2]] = other.WF[
                    :, i + s * other.ig.shape[1]
                ]
            res += np.einsum("mabc,nabc->mn", WF1.conj(), WF2)
        #        return np.einsum("mabc,nabc->mn",WF1.conj(),WF2)
        #        return np.einsum("ma,na->mn",self.WF.conj(),other.WF)
        return res

    def getloc1(self, loc):
        gmax = abs(self.ig[:3]).max(axis=1)
        grid = [np.linspace(0.0, 1.0, 2 * gm + 1, False) for gm in gmax]
        print("grid:", grid)
        loc_grid = loc(
            grid[0][:, None, None], grid[1][None, :, None], grid[2][None, None, :]
        )
        print("loc=", loc, "loc_grid=\n", loc_grid)
        #        FFTgrid=np.zeros( (self.Nband,*(2*gmax+1)),dtype=complex )
        res = np.zeros(self.Nband)
        for s in [0, 1] if self.spinor else [0]:
            WF1 = np.zeros((self.Nband, *(2 * gmax + 1)), dtype=complex)
            for i, ig in enumerate(self.ig.T):
                WF1[:, ig[0], ig[1], ig[2]] = self.WF[:, i + s * self.ig.shape[1]]
            #            print ("wfsum",WF1.sum()," shape ",WF1.shape,loc_grid.shape)
            res += np.array(
                [
                    np.sum(np.abs(np.fft.ifftn(WF1[ib])) ** 2 * loc_grid).real
                    for ib in range(self.Nband)
                ]
            )
        print("    ", loc_grid.shape)
        return res * (np.prod(loc_grid.shape))

    def getloc(self, locs):
        return np.array([self.getloc1(loc) for loc in locs])
