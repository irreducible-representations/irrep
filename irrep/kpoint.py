
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


from functools import lru_cache
import numpy as np
import numpy.linalg as la
import copy
from .gvectors import symm_eigenvalues, symm_matrix, get_pw_energies
from .utility import cached_einsum, compstr, get_block_indices, is_round, format_matrix, log_message, orthogonalize, vector_pprint


class Kpoint:
    """
    Organizes info about the states and energy-levels of a 
    particular k-point in attributes. Contains methods to calculate and write 
    traces (and irreps), for the separation of the band structure in terms of a 
    symmetry operation and for the calculation of the Zak phase.

    Parameters
    ----------
    ik : int
        Index of kpoint, starting count from 0.
    NBin : int
        Number of bands considered at the k-point in the DFT calculation.
    Ecut : float
        Plane-wave cutoff (in eV) to consider in the expansion of wave-functions.
    Ecut0 : float
        Plane-wave cutoff (in eV) used in the DFT calulation. Always read from 
        DFT files. Insignificant if `code`='wannier90'.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
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
    eKG : array, shape=(ng, dtype=float)
        the energies of the plane-waves in the expansion of wave-functions.


    Attributes
    ----------
    spinor : bool
        `True` if wave-functions are spinors, `False` if they are scalars.
    ik0 : int
        Index of the k-point, starting the count from 1.
    num_bands : int
        Number of bands whose traces should be calculated.
    spinor : bool   
        `True` if wave-functions are spinors, `False` if they are scalars.
    nspinor : int
        Number of spinor components in the wave functions. Returns 2 for spinors, 1 for scalars.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    WF : array( (num_bands, NG, nspinor), dtype=complex)
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
    k : array, shape=(3,)
        Direct coordinates of the k point in the DFT cell setting.
    K : array, shape=(3,)
        Property getter for `self.k`. NEEDED TO PRESERVE COMPATIBILITY WITH
        BANDUPPY<=0.3.3. DO NOT CHANGE UNLESS NECESSARY. NOTIFY THE DEVELOPERS 
        IF ANY CHANGES ARE MADE.
    k_refUC : array, shape=(3,)
        Direct coordinates of the k point in the reference cell setting.
    Energy_raw : array
        energy levels of each state (1 energy for every row in WF)
    Energy_mean : array
        Energy-levels of degenerate groups of bands whose traces should be calculated.
    upper : float
        Energy of the first band above the set of bands whose traces should be 
        calculated. It will be set to `numpy.NaN` if the last band matches the 
        last band in the DFT calculation (default `IBend`).
    char : array
        Each row corresponds to a set of degenerate states. Each column is the 
        trace of a symmetry in the little cogroup in the DFT cell setting.
    char_refUC : array
        The same as `char`, but in the reference cell setting.
    degeneracies : array
        Degeneracies of energy levels between `IBstart` and `IBend`.
    block_indices : array((N,2), dtype=int)
        Integers representing the band index of the first and last+1  state in each set of 
        degenerate states. The bounds can be obtained as 
        `for ibot, itop in block_indices: Emean=Energy[ibot:itop].mean()`
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
        num_bands=None,
        RecLattice=None,  # this was last mandatory argument
        spinor=None,
        kpt=None,
        WF=None,  # first arg added for abinit (to be kept at the end)
        Energy=None,
        ig=None,
        upper=None,
        normalize=True,
        eKG=None,
    ):

        if spinor is None:
            if WF is not None:
                spinor = (WF.shape[2] == 2)  # if WF is provided, check the number of components
            else:
                raise ValueError("spinor must be specified if WF is not provided")
        else:
            if WF is not None:
                if WF.shape[2] != (2 if spinor else 1):
                    raise ValueError(
                        f"WF must have {2 if spinor else 1} components if spinor is {spinor}, got {WF.shape[1]} components"
                    )
        self.spinor = spinor

        if ik is None:
            self.ik0 = None
        else:
            self.ik0 = ik + 1  # the index in the WAVECAR (count start from 1)

        if num_bands is None:
            if Energy is not None:
                num_bands = len(Energy)
            elif WF is not None:
                num_bands = WF.shape[0]
            else:
                raise ValueError("num_bands must be specified if neither WF nor Energy are not provided")

        self.num_bands = num_bands
        self.RecLattice = RecLattice
        self.upper = upper

        self.k = kpt
        self.WF = WF
        self.Energy_raw = Energy
        self.ig = ig
        eKGcalc = self.calc_egk()
        if eKG is None:
            eKG = eKGcalc
        else:
            assert np.allclose(eKG, eKGcalc, atol=1e-4), f"eKG provided {eKG} does not match calculated {eKGcalc}, ration is {eKG / eKGcalc}"
            # pass
        self.eKG = eKG
        self.upper = upper

        # self.k_refUC = np.dot(refUC.T, self.k) % 1

        if normalize:
            self.WF /= (
                np.sqrt(np.abs(cached_einsum("ijs,ijs->i", self.WF.conj(), self.WF)))
            ).reshape(self.num_bands, 1, 1)
            # np.linalg.norm(self.WF, axis=(1,2))[:, None, None]

    def set_little_group(self, symmetries):
        """
        Set the little group of the k-point based on the provided symmetries.
        Parameters
        ----------
        symmetries : list
            List of symmetry operations (instances of `SymmetryOperation`)

        Sets the `little_group` attribute, which contains the symmetry operations
        that leave the k-point invariant up to a reciprocal lattice vector.
        """
        self.little_group = []
        for symop in symmetries:
            k_rotated = np.dot(np.linalg.inv(symop.rotation).T, self.k)
            dkpt = np.round(k_rotated - self.k)
            if np.allclose(dkpt, k_rotated - self.k):
                self.little_group.append(symop)
        return self.little_group

    def calc_egk(self):
        return get_pw_energies(self.RecLattice, self.k, self.ig)

    @property
    def nspinor(self):
        """
        Number of spinor components in the wave functions.
        Returns 2 for spinors, 1 for scalars.
        """
        return 2 if self.spinor else 1

    def init_traces(self, degen_thresh=1e-8, verbosity=0, calculate_traces=True, refUC=np.eye(3), shiftUC=np.zeros(3),
                    symmetries_tables=None, save_wf=True):
        """
        Continuation of __init__ method. Calculates traces of symmetry eigenvalues and irreps, when asked. 
        Separated because it is used in the `copy_sub` method.

        Parameters
        ----------
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
        calculate_traces : bool
            If `True`, traces of symmetries will be calculated. Useful to create 
            instances faster.
        save_wf : bool
            Whether wave functions should be kept as attribute after calculating 
            traces.
        verbosity : int
            Verbosity level. Default set to minimalistic printing
        """

        self.k_refUC = np.dot(refUC.T, self.k)

        # Sort symmetries based on their indices
        argsort = np.argsort([symop.ind for symop in self.little_group])
        self.little_group = [self.little_group[ind] for ind in argsort]

        # Determine degeneracies
        self.block_indices = get_block_indices(self.Energy_raw, thresh=degen_thresh, cyclic=False)
        self.degeneracies = self.block_indices[:, 1] - self.block_indices[:, 0]

        # Calculate traces
        if calculate_traces:
            self.char, self.char_refUC, self.Energy_mean = \
                self.calculate_traces(refUC, shiftUC, symmetries_tables, verbosity, use_blocks=False)

            # Determine number of band inversions based on parity
            found = False
            for i, sym in enumerate(self.little_group):
                if (
                    sum(abs(sym.translation)) < 1e-6 and
                    abs(sym.rotation + np.eye(3)).sum() < 1e-6
                ):
                    found = True
                    break
            if found:
                # Number of inversion odd states (not pairs!)
                print(f"inversion is {i}-th symmetry in the little group")
                print(f"characters of inversion symmetry: {self.char[:, i]}")
                print(f"degeneracies: {self.degeneracies}")
                self.num_bandinvs = int(round(sum(self.degeneracies - self.char[:, i].real) / 2))
            else:
                self.num_bandinvs = None
            print(f"number of inversion-odd states: {self.num_bandinvs}")

        if not save_wf:
            self.WF = None

    @property
    def k_cart(self):
        return np.dot(self.k, self.RecLattice)

    @property
    def ig_cart(self):
        """
        Returns the ig vectors in cartesian coordinates.
        """
        return np.dot(self.ig[:, :3], self.RecLattice)

    @property
    def K(self):
        """Getter for the redfuced coordinates of the k-point
        needed to keep compatibility with banduppy

        ACCESSED BY BANDUPPY.NEEDED TO PRESERVE COMPATIBILITY WITH 
        BANDUPPY<=0.3.3. AVOID CHANGING UNLESS NECESSARY.
        """
        return self.k

    def k_close_mod1(self, kpt, prec=1e-6):
        """
        Check if the k-point is close to another k-point modulo 1. (in reduced 
        coordinates)

        ACCESSED BY BANDUPPY. AVOID CHANGING UNLESS NECESSARY. NOTIFY 
        DEVELOPERS IF ANY CHANGE IS MADE.

        Parameters
        ----------
        kpt : array
            Coordinates of the k-point to compare.
        prec : float, default=1e-6
            Threshold to consider the k-points as equal.
        """
        return is_round(self.k - kpt, prec=prec)

    def copy_sub(self, E, WF, kwargs_kpoint={}):
        """
        Create an instance of class `Kpoint` for a restricted set of states.

        Parameters
        ----------
        E : array
            Energy-levels of states.
        WF : array
            Coefficients of the plane-wave expansion of wave-functions. Each row 
            corresponds to a wave-function, each column to a plane-wave.
        kwargs_kpoint : dict, optional
            Additional keyword arguments to pass to the `init_traces` method,
        Returns
        -------
        other : class
            Instance of `Kpoints` corresponding to the group of states passed. 
            They are shorted by energy-levels.
        """
        other = copy.copy(self)  # copy of whose class
        # Sort energy levels
        sortE = np.argsort(E)
        other.Energy_raw = E[sortE]
        other.WF = WF[sortE]
        other.num_bands = len(E)
        if kwargs_kpoint is not None:
            other.init_traces(**kwargs_kpoint)
            other.identify_irreps()
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
        if not self.k_close_mod1(kptPBZ.dot(supercell.T), prec=1e-5):
            raise RuntimeError(f"unable to unfold {self.k} to {kptPBZ}, with supercell={supercell}")
        g_shift = kptPBZ - self.k.dot(np.linalg.inv(supercell.T))
        selectG = np.array(
            np.where(
                [
                    is_round(dg, prec=1e-4)
                    for dg in (self.ig[:3].T.dot(np.linalg.inv(supercell.T)) - g_shift)
                ]
            )[0]
        )
        if self.spinor:
            selectG = np.hstack((selectG, selectG + self.NG))
        WF = self.WF[:, selectG, :]
        result = []
        for b1, b2, E, matrices in self.get_rho_spin(degen_thresh):
            # proj = np.array(
            #     [
            #         [WF[i].reshape(-1).conj().dot(WF[j].reshape(-1)) for j in range(b1, b2)]
            #         for i in range(b1, b2)
            #     ]
            # )
            proj = cached_einsum('igs,jgs->ij', WF[b1:b2].conj(), WF[b1:b2])
            result.append([E,] + [np.trace(proj.dot(M)).real for M in matrices])
        return np.array(result)

    @property
    def NG(self):
        """Getter for the number of plane-waves in current k-point"""
        return self.ig.shape[0]

    @lru_cache
    def get_rho_spin(self, degen_thresh=1e-4):
        r"""
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
        block_indices = get_block_indices(self.Energy_raw, thresh=degen_thresh)
        result = []
        for b1, b2 in block_indices:
            E = self.Energy_raw[b1:b2].mean()
            W = cached_einsum('igs,jgs->ij', self.WF[b1:b2].conj(), self.WF[b1:b2])
            # np.array( [ [self.WF[i].conj().dot(self.WF[j])
            #                  for j in range(b1, b2)] for i in range(b1, b2)])
            if self.spinor:
                # ng = self.NG
                # [
                #     [ np.array( [
                #         [self.WF[i, ng * s: ng * (s + 1)].conj().dot(
                #             self.WF[j, ng * t: ng * (t + 1)]) for j in range(b1, b2)]
                #                 for i in range(b1, b2)] )# band indices
                #         for t in (0, 1) ]
                #         for s in (0, 1) ]  # spin indices
                Smatrix = cached_einsum('igs,jgt->ijst', self.WF[b1:b2].conj(), self.WF[b1:b2])
                Sx = Smatrix[0][1] + Smatrix[1][0]
                Sy = 1j * (-Smatrix[0][1] + Smatrix[1][0])
                Sz = Smatrix[0][0] - Smatrix[1][1]
                result.append((b1, b2, E, (W, Sx, Sy, Sz)))
            else:
                result.append((b1, b2, E, (W,)))
        return result

    def normWF(self):
        return np.linalg.norm(self.WF, axis=(1, 2))

    def Separate(self, symop, groupKramers=True, verbosity=0,
                 kwargs_kpoint={}):
        """
        Separate the band structure in a particular k-point according to the 
        eigenvalues of a symmetry operation.

        Parameters
        ----------
        isymop : int
            Index of symmetry used for the separation.
        groupKramers : bool, default=True
            If `True`, states will be coupled by pairs of Kramers.
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing

        Returns
        -------
        subspaces : dict
            Each key is an eigenvalue of the symmetry operation and the
            corresponding value is an instance of `class` `Kpoint` for the 
            states with that eigenvalue.
        """

        # Check orthogonality of wave functions
        # Rm once tests are fixed
        norms = self.normWF()**2
        check = np.max(abs(norms - np.eye(norms.shape[0])))
        if check > 1e-5:
            log_message(f"orthogonality (largest of diag. <psi_nk|psi_mk>): {check:7.5f} > 1e-5   \n",
                        verbosity, 1)


        S = symm_matrix(
            K=self.k,
            WF=self.WF,
            igall=self.ig,
            A=symop.rotation,
            S=symop.spinor_rotation,
            T=symop.translation,
            spinor=self.spinor,
        )


        # Check that S is block-diagonal
        Sblock = np.copy(S)
        for b1, b2 in self.block_indices:
            Sblock[b1:b2, b1:b2] = 0
        check = np.max(abs(Sblock))
        if check > 0.001:
            log_message("WARNING: matrix of symmetry has non-zero elements between "
                        f"states of different energy:  {check} \n", verbosity, 1)
            log_message(f"Printing matrix of symmetry at k={self.k}", verbosity, 1)
            log_message(format_matrix(Sblock), verbosity, 1)
            log_message("The diagonal blocks", verbosity, 1)
            log_message(", ".join([f"{b1}:{b2} \n: {format_matrix(S[b1:b2, b1:b2])}" for b1, b2 in self.block_indices]), verbosity, 1)


        # Calculate eigenvalues and eigenvectors in each block
        eigenvalues = []
        eigenvectors = []
        Eloc = []
        for istate, (b1, b2) in enumerate(self.block_indices):
            S_loc = orthogonalize(S[b1:b2, b1:b2], verbosity=verbosity, error_threshold=1e-2, warning_threshold=1e-3)
            W, V = la.eig(S_loc)
            for w, v in zip(W, V.T):
                eigenvalues.append(w)
                Eloc.append(self.Energy_mean[istate])
                eigenvectors.append(
                    np.hstack((np.zeros(b1), v, np.zeros(self.num_bands - b2)))
                )
        w = np.array(eigenvalues)
        v = np.array(eigenvectors).T  # each col an eigenvector
        Eloc = np.array(Eloc)

        # Check unitarity of the symmetry
        if np.abs((np.abs(w) - 1.0)).max() > 1e-4:
            log_message(f"WARNING: some eigenvalues are not unitary: {w}", verbosity, 1)
        if np.abs((np.abs(w) - 1.0)).max() > 3e-1:
            raise RuntimeError(f" some eigenvalues are not unitary :{w} ")
        w /= np.abs(w)

        subspaces = {}

        if groupKramers:

            # Sort based on real part of eigenvalues
            arg = np.argsort(np.real(w))
            w = w[arg]
            v = v[:, arg]
            Eloc = Eloc[arg]
            block_indices = get_block_indices(w, thresh=0.05, cyclic=False)

            for b1, b2 in block_indices:
                v1 = v[:, b1:b2]
                subspaces[w[b1:b2].mean()] = self.copy_sub(E=Eloc[b1:b2],
                                                           WF=cached_einsum('ij,jks->iks', v1.T, self.WF),
                                                           kwargs_kpoint=kwargs_kpoint)

        else:  # don't group Kramers pairs

            # Sort based on the argument of eigenvalues
            # arg = np.argsort( (np.angle(w)/(2*np.pi)+0.01)%1 ) # to make sure that we start from the +1 and go anti-clockwise
            arg = np.argsort(np.angle(w))
            w = w[arg]
            v = v[:, arg]
            Eloc = Eloc[arg]
            block_indices = get_block_indices(w, thresh=0.05, cyclic=True)

            for b1, b2 in block_indices:
                v1 = np.roll(v, -b1, axis=1)[:, : (b2 - b1) % self.num_bands]
                subspaces[np.roll(w, -b1)[: (b2 - b1) % self.num_bands].mean()] = self.copy_sub(
                    E=np.roll(Eloc, -b1)[: (b2 - b1) % self.num_bands],
                    WF=cached_einsum('ij,jks->iks', v1.T, self.WF),
                    kwargs_kpoint=kwargs_kpoint
                )

        return subspaces

    def symm_matrix(self, other, symop, block_indices=None, unitary=True, unitary_params={}, Ecut=None):
        K1 = self
        K2 = other
        return symm_matrix(
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
            Ecut=Ecut,
            eKG=K1.eKG
        )

    def calculate_traces(self, refUC, shiftUC, symmetries_tables, verbosity=0, use_blocks=True):
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

        Returns
        -------
        char : array
            Each row corresponds to a set of degenerate states. Each column is the 
            trace of a symmetry in the little cogroup in the DFT cell setting.
        char_refUC : array
            The same as `char`, but in the reference cell setting.
        Energy_mean : array
            Average of energy levels within each set of degenerate states
        '''

        # Put all traces in an array. Rows (cols) correspond to syms (wavefunc)
        char = []
        for symop in self.little_group:
            char.append(
                symm_eigenvalues(
                    K=self.k,
                    WF=self.WF,
                    igall=self.ig,
                    A=symop.rotation,
                    S=symop.spinor_rotation,
                    T=symop.translation,
                    spinor=self.spinor,
                    block_ind=self.block_indices if use_blocks else None
                ))
        char = np.array(char)

        log_message(f"char.shape = {char.shape}, Energy_raw.shape = {self.Energy_raw.shape}, block_indices = {self.block_indices}", verbosity, 2)

        # Check that number of irreps is int
        Nirrep = np.linalg.norm(char.sum(axis=1)) ** 2 / char.shape[0]
        if abs(Nirrep - round(Nirrep)) > 1e-2:
            log_message(f"WARNING - non-integer number of states : {Nirrep}", verbosity, 2)
        Nirrep = int(round(Nirrep))

        # Sum traces of degenerate states. Rows (cols) correspond to states (syms)
        char = np.array(
            [char[:, start:end].sum(axis=1) for start, end in self.block_indices]
        )

        # Take average of energies over degenerate states
        Energy_mean = np.array(
            [self.Energy_raw[start:end].mean() for start, end in self.block_indices]
        )

        # Transfer traces in calculational cell to refUC
        char_refUC = char.copy()
        if (not np.allclose(refUC, np.eye(3, dtype=float)) or
                not np.allclose(shiftUC, np.zeros(3, dtype=float))):
            # Calculational and reference cells are not identical
            for i, sym in enumerate(self.little_group):
                dt = (symmetries_tables[sym.ind - 1].t -
                      sym.translation_refUC(refUC, shiftUC))
                char_refUC[:, i] *= (sym.sign *
                                     np.exp(-2j * np.pi * dt.dot(self.k_refUC)))

        return char, char_refUC, Energy_mean


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
            values are traces of symmetries. Traces are in DFT cell.
        '''
        if irreptable is None:
            if hasattr(self, 'irreptable'):
                irreptable = self.irreptable
        self.irreptable = irreptable

        self.onlytraces = irreptable is None
        if self.onlytraces:
            # irreps = ["None"] * (len(self.degeneracies) - 1)  # here was a -1, IDK why
            irreps = ["None"] * (len(self.degeneracies))  # removed the -1

        else:

            # irreps is a list. Each element is a dict corresponding to a
            # group of degen. states. Every key is an irrep and its value
            # the multiplicity of the irrep in the rep. of degen. states
            try:
                irreps = []
                for ch in self.char:
                    multiplicities = {}
                    for ir in irreptable:
                        ir_characters = np.array([irreptable[ir][sym.ind] for sym in self.little_group])
                        # some coreps are not normalized
                        # this is len(ch) for all irreps
                        normalization = (np.abs(ir_characters) ** 2).sum()
                        multipl = np.dot(ir_characters, ch.conj()) / normalization
                        if abs(multipl) > 1e-3:
                            multiplicities[ir] = multipl
                    irreps.append(multiplicities)
            except KeyError as ke:
                print(ke)
                print("irreptable:", irreptable)
                print([sym.ind for sym in self.little_group])
                raise ke

        self.irreps = irreps

    def copy(self):
        """
        Create a copy of the current k-point instance.

        Returns
        -------
        Kpoint
            A new instance of `Kpoint` with the same attributes as the current one.
        """
        return Kpoint(ik=-1,
                      RecLattice=self.RecLattice,
                      spinor=self.spinor,
                      kpt=self.k.copy(),
                      WF=self.WF.copy(),  # first arg added for abinit (to be kept at the end)
                      Energy=self.Energy_raw.copy(),
                      ig=self.ig.copy(),
                      upper=self.upper,
                      normalize=False,  # already normalized in the original instance (if needed)
                        )




    def get_transformed_copy(self, symmetry_operation, k_new=None):
        """
        Get a copy of the k-point transformed by a symmetry operation.

        Parameters
        ----------
        symmetry_operation : SymmetryOperation
            Symmetry operation to apply to the k-point.
        Returns
        -------
        Kpoint
            A new instance of `Kpoint` with the k-point transformed by the
            symmetry operation.
        """
        _k, _WF, _ig = symmetry_operation.transform_WF(k=self.k, WF=self.WF, igall=self.ig, k_new=k_new)
        return Kpoint(ik=self.ik0,
                      RecLattice=self.RecLattice,
                      spinor=self.spinor,
                      kpt=_k,
                      WF=_WF,  # first arg added for abinit (to be kept at the end)
                      Energy=self.Energy_raw.copy(),
                      ig=_ig,
                      upper=self.upper,
                      normalize=False,  # already normalized in the original instance (if needed)
                      eKG=self.eKG.copy()
                      )

    def write_characters(self):
        '''
        Write the block of data of the k point, including energy levels,
        degeneracies, traces and irreps.
        '''

        # Print header for k-point
        print(f"\n\n k-point {self.ik0:3d} : {vector_pprint(np.round(self.k, 5))} (in DFT cell)\n"
              f"               {vector_pprint(np.round(self.k_refUC, 5))} (after cell trasformation)\n\n"
              f" number of states : {self.num_bands}\n"
              )

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
                    s += f'({irreps[ir].real:.5}'
                    if abs(irreps[ir].imag) > 1e-4:
                        s += f'{irreps[ir].imag:+.5f}i'
                    s += ')'
            str_irreps.append(s)

        # Set auxiliary blank strings for formatting
        writeimaginary = np.abs(self.char.imag).max() > 1e-4
        if writeimaginary:
            aux1 = ' ' * 4
        else:
            aux1 = ''
        irreplen = max(len(irr) for irr in str_irreps)
        # if irreplen % 2 == 1:
        #    irreplen += 1
        # aux2 = " " * int(irreplen / 2 - 3)
        num_spaces = (irreplen - 8) / 2
        aux2 = " " * int(num_spaces)
        if irreplen % 2 == 0:
            aux3 = aux2
        else:
            aux3 = aux2 + " "

        print(f"   Energy  |   degeneracy  | {aux2} irreps {aux3} | sym. operations  ")

        # Print indices of little-group symmetries
        s = f"           |               | {aux2}        {aux3} | "
        inds = []
        for sym in self.little_group:
            inds.append(f"{aux1}{sym.ind:4d}    {aux1}")
        s += " ".join(inds)
        print(s)

        # Print line associated to a set of degenerate states
        for e, d, ir, ch1, ch2 in zip(self.Energy_mean, self.degeneracies, str_irreps, self.char, self.char_refUC):
            # Traces in DFT unit cell
            right_str1 = []
            right_str2 = []
            for tr1, tr2 in zip(ch1, ch2):
                s1 = f"{tr1.real:8.4f}"
                s2 = f"{tr2.real:8.4f}"
                if writeimaginary:
                    s1 += f"{tr1.imag:+7.4f}j"
                    s2 += f"{tr2.imag:+7.4f}j"
                right_str1.append(s1)
                right_str2.append(s2)
            right_str1 = ' '.join(right_str1)
            right_str2 = ' '.join(right_str2)

            # Energy, degeneracy, irrep's label and character in DFT cell
            left_str = f" {e:8.4f}  |    {d:5d}      | {ir:{irreplen}s} |"
            print(left_str + " " + right_str1)

            # Line for character in reference cell
            left_str = f"           |               | {' ' * len(ir):{irreplen}s} |"
            print(left_str + " " + right_str2)  # line for character in DFT


    def json(self):
        '''
        Prepare the data to save it in JSON format.

        Returns
        -------
        json_data : dict
            Data that will be saved in JSON format.
        '''

        json_data = {}
        json_data['k'] = self.k
        json_data['k_refUC'] = self.k_refUC

        indices_symmetries = [sym.ind for sym in self.little_group]
        json_data['symmetries'] = list(indices_symmetries)

        # Energy levels and degeneracies
        json_data['energies_mean'] = self.Energy_mean
        json_data['energies_raw'] = self.Energy_raw
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
        json_data['characters refUC'] = self.char_refUC
        if np.allclose(self.char, self.char_refUC, rtol=0.0, atol=1e-4):
            json_data['characters refUC is the same'] = True
        else:
            json_data['characters refUC is the same'] = False

        return json_data

    def write_irrepsfile(self, file):
        '''
        Write the irreps of this k point into `irreps.dat` file.

        Parameters
        ----------
        file : File object
            File object for the `irreps.dat` file.
        '''

        for energy, irrep_dict in zip(self.Energy_mean, self.irreps):
            irrep = ''.join(irrep_dict.keys())
            s = f'{energy:15.7f}    {irrep:15s}\n'
            file.write(s)


    def write_plotfile(self, kpl, efermi):

        writeimaginary = np.abs(self.character.imag).max() > 1e-4
        s = []
        for e, dim, char in zip(self.Energy_mean, self.degeneracies, self.character):
            s_loc = f'{kpl:8.4f}   {e - efermi:8.4f}      {dim:5d}   '
            for tr in char:
                s_loc += f"{tr.real:8.4f}"
                if writeimaginary:
                    s_loc += f"{tr.imag:+7.4f}j "
            s.append(s_loc)
        s = '\n'.join(s)
        s += '\n\n'


    def write_irrepfile(self, firrep):

        file = open(firrep, "a")
        for e, ir in zip(self.Energy_mean, self.irreps):
            for irrep in ir.split(","):
                try:
                    weight = abs(compstr(irrep.split("(")[1].strip(")")))
                    if weight > 0.3:
                        file.write(f" {irrep.split('(')[0]:10s}   {e:10.5f}\n")
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

        # Line 1: order of the little cogroup
        # Line 2: indices of syms in the little cogroup
        # Line 3: for each band introduce a row with the followind data:
        # (1) 1+number of bands below, (2) dimension (degeneracy) of the band,
        # (3) energy and eigenvalues (real part, imaginary part) for each
        # symmetry operation of the little group (listed above).
        indices = [symop.ind for symop in self.little_group]
        res = (f"{len(self.little_group)} \n {'  '.join(str(x) for x in indices)} \n")

        IB = np.cumsum(np.hstack(([0], self.degeneracies[:-1]))) + 1
        res += (
            "\n".join(
                f" {ib:8d}  {d:8d}   {e:8.4f} " + "  ".join(f"{c.real:10.6f}   {c.imag:10.6f} " for c in ch)
                for e, d, ib, ch in zip(self.Energy_mean, self.degeneracies, IB, self.char)
            ) +
            "\n"
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
        assert self.spinor == other.spinor, "Spinor property of k-points should be the same"
        g = np.array((self.k - other.k).round(), dtype=int)
        igall = np.vstack((self.ig[:, :3], other.ig[:, :3] - g[None, :]))
        igmax = igall.max(axis=0)
        igmin = igall.min(axis=0)
        igsize = igmax - igmin + 1
        #        print (self.ig.T)
        #        print (igsize)
        res = np.zeros((self.num_bands, other.num_bands), dtype=complex)

        # short again coefficients of expansions
        # for s in [0, 1] if self.spinor else [0]:
        WF1 = np.zeros((self.num_bands, igsize[0], igsize[1], igsize[2], self.nspinor), dtype=complex)
        WF2 = np.zeros((other.num_bands, igsize[0], igsize[1], igsize[2], self.nspinor), dtype=complex)
        for i, ig in enumerate(self.ig.T):
            WF1[:, ig[0] - igmin[0], ig[1] - igmin[1], ig[2] - igmin[2], :] = self.WF[:, i, :]
        for i, ig in enumerate(other.ig[:3].T - g[None, :]):
            WF2[:, ig[0] - igmin[0], ig[1] - igmin[1], ig[2] - igmin[2]] = other.WF[:, i, :]
        res += cached_einsum("mabcs,nabcs->mn", WF1.conj(), WF2)
        return res

    # I think these routines are not used anymore, but I leave them here for reference
    # def getloc1(self, loc):
    #     gmax = abs(self.ig[:3]).max(axis=1)
    #     grid = [np.linspace(0.0, 1.0, 2 * gm + 1, False) for gm in gmax]
    #     print("grid:", grid)
    #     loc_grid = loc(
    #         grid[0][:, None, None], grid[1][None, :, None], grid[2][None, None, :]
    #     )
    #     print("loc=", loc, "loc_grid=\n", loc_grid)
    #     res = np.zeros(self.num_bands)
    #     WF1 = np.zeros((self.num_bands, *(2 * gmax + 1)), dtype=complex)
    #     for s in [0, 1] if self.spinor else [0]:
    #         WF1 = np.zeros((self.num_bands, *(2 * gmax + 1)), dtype=complex)
    #         for i, ig in enumerate(self.ig.T):
    #             WF1[:, ig[0], ig[1], ig[2]] = self.WF[:, i + s * self.ig.shape[1]]
    #         #            print ("wfsum",WF1.sum()," shape ",WF1.shape,loc_grid.shape)
    #         res += np.array([np.sum(np.abs(np.fft.ifftn(WF1[ib])) ** 2 * loc_grid).real
    #                          for ib in range(self.num_bands)])
    #     print("    ", loc_grid.shape)
    #     return res * (np.prod(loc_grid.shape))

    # def getloc(self, locs):
    #     return np.array([self.getloc1(loc) for loc in locs])
