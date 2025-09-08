
import warnings
from .utility import log_message, orthogonalize, cached_einsum
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


import numpy as np
import numpy.linalg as la
Rydberg_eV = 13.605693  # eV
Hartree_eV = 2 * Rydberg_eV
bohr_angstrom = 0.52917721092  # Angstrom


class NotSymmetryError(RuntimeError):
    """
    Pass if we attemp to apply to a k-vector a symmetry that does not belong 
    to its little-group.
    """
    pass


#  constant  below is 2m/hbar**2 in units of 1/eV Ang^2 (value is
#  adjusted in final decimal places to agree with VASP value; program
#  checks for discrepancy of any results between this and VASP values)
twomhbar2 = 0.262465831


def get_pw_energies(RecLattice, k, ig):
    """
    Calculates the plane-wave energies at a given k-point.

    Parameters
    ----------
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    k : array, shape=(3,)
        Direct coordinates of the k-point.
    ig : array, shape=(n, 3)
        Each row contains the integer coefficients of a reciprocal lattice 
        vector taking part in the plane-wave expansion of wave-functions at 
        the current k-point.

    Returns
    -------
    eKG : array
        Energies of the plane-waves at the given k-point.
    """
    return 0.5 * Hartree_eV * (bohr_angstrom**2) * la.norm((k[None, :] + ig[:, :3]) @ RecLattice, axis=1) ** 2


# This function is a python translation of a part of WaveTrans Code
def calc_gvectors(
    K,
    RecLattice,
    Ecut,
    nplane=np.inf,
    Ecut1=-1,
    thresh=1e-3,
    spinor=True,
    nplanemax=10000,
    verbosity=0
):
    """ 
    Generates G-vectors taking part in the plane-wave expansion of 
    wave-functions in a particular k-point. Optionally, a cutoff `Ecut1` is 
    applied to get rid of large G-vectors.

    Parameters
    ----------
    K : array
        Direct coordinates of the k-point.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    Ecut : float
        Plane-wave cutoff (in eV) used in the DFT calulation. Always read from 
        DFT files.
    nplane : int, default=np.inf
        Number of plane-waves in the expansion of wave-functions (read from DFT 
        files). Only significant for VASP. 
    Ecut1 : float, default=Ecut
        Plane-wave cutoff (in eV) to consider in the expansion of wave-functions.
    thresh : float, default=1e-3
        Threshold for defining the g-vectors with the same energy.
    spinor : bool, default=True
        `True` if wave functions are spinors, `False` if they are scalars. It 
        will be read from DFT files. Mandatory for `vasp`.
    nplanemax : int, default=10000
        Sets the maximun number of iterations when calculating vectors.
    verbosity : int, default=0
        Level of verbosity. If 0, no output is printed. If 1, only the most 
        important messages are printed. If 2, all messages are printed.

    Returns
    -------
    igall : array
        Every row corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of columns is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth column stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) column contains the index of the first (last) plane-wave with 
        the same energy as the plane-wave of the current column.

"""

    log_message(f"Generating plane waves at k: ({' '.join(f'{x:6.3f}' for x in K)})", verbosity, 2)
    if Ecut1 <= 0:
        Ecut1 = Ecut
    B = RecLattice

    igp = np.zeros(3)
    igall = []
    Eg = []
    memory = np.full(10, True)
    for N in range(nplanemax):
        flag = True
        if N % 10 == 0:
            log_message(f'Cycle {N:>3d}: number of plane waves = {len(igall):>10d}', verbosity, 2)
        if len(igall) >= nplane:     # Only enters if vasp
            log_message(f"{N=}, {memory=}", verbosity, 2)
            if np.all(memory):  # probably spinor wrong set as spinor=F
                raise RuntimeError(
                    "calc_gvectors is stuck calculating plane waves of energy larger "
                    f"than cutoff Ecut = {Ecut}. Make sure that the "
                    "VASP calculation does not include SOC and set -spinor if it does."
                )
            else:
                log_message(
                    f"Reached the maximum number of plane-waves {nplane} at N={N}, "
                    "stopping the calculation", verbosity, 2
                )
                break

        for ig3 in range(-N, N + 1):
            for ig2 in range(-(N - abs(ig3)), N - abs(ig3) + 1):
                for ig1 in set([-(N - abs(ig3) - abs(ig2)), N - abs(ig3) - abs(ig2)]):
                    igp = (ig1, ig2, ig3)
                    etot = la.norm((K + np.array(igp)).dot(B)) ** 2 / twomhbar2
                    if etot < Ecut:
                        igall.append(igp)
                        Eg.append(etot)
                        flag = False
        memory[:-1] = memory[1:]
        memory[-1] = flag

    ncnt = len(igall)
    if nplane < np.inf:  # vasp
        if ncnt != nplane:
            raise RuntimeError(f"*** error - computed ncnt={ncnt} != input nplane={nplane}")
    igall = np.array(igall, dtype=int)
    ng = igall.max(axis=0) - igall.min(axis=0)
    igall1 = igall % ng[None, :]
    igallsrt = np.argsort((igall1[:, 2] * ng[1] + igall1[:, 1]) * ng[0] + igall1[:, 0])
    igall1 = igall[igallsrt]
    Eg = np.array(Eg)[igallsrt]
    igall = np.zeros((ncnt, 6), dtype=int)
    igall[:, :3] = igall1
    igall[:, 3] = np.arange(ncnt)
    igall = igall[Eg <= Ecut1]
    Eg = Eg[Eg <= Ecut1]
    srt = np.argsort(Eg)
    Eg = Eg[srt]
    igall = igall[srt, :]
    wall = [0] + list(np.where(Eg[1:] - Eg[:-1] > thresh)[0] + 1) + [igall.shape[0]]
    for i in range(len(wall) - 1):
        igall[wall[i]: wall[i + 1], 4] = wall[i]
        igall[wall[i]: wall[i + 1], 5] = wall[i + 1]
    return igall, Eg


def sortIG(ik, kg, kpt, WF, RecLattice, Ecut0, Ecut, verbosity=0):
    """
    Apply plane-wave cutoff specified in CLI to the expansion of 
    wave-functions and sort the coefficients and plane-waves in ascending 
    order in energy.

    Parameters
    ----------
    ik : int
        Index of the k-point.
    kg : array
        Each row contains the integer coefficients of a reciprocal lattice 
        vector taking part in the plane-wave expansion of wave-functions at 
        the current k-point.
    kpt : array, shape=(3,)
        Direct coordinates of the k-point.
    WF : array
        `WF[i,j]` contains the complex coefficient corresponding to 
        :math:`j^{th}` plane-wave in the expansion of :math:`i^{th}` 
        wave-function.
    RecLattice : array, shape=(3,3)
        Each row contains the cartesian coordinates of a basis vector forming 
        the unit-cell in reciprocal space.
    Ecut : float
        Plane-wave cutoff (in eV) to consider in the expansion of wave-functions.
        Will be set equal to `Ecut0` if input parameter `Ecut` was not set or 
        the value of this is negative or larger than `Ecut0`.
    Ecut0 : float
        Plane-wave cutoff (in eV) used for DFT calulations. Always read from 
        DFT files. Insignificant if `code`=`wannier90`.
    spinor : bool
        `True` if wave-functions are spinors, `False` if they are scalars.

    Returns
    -------
    WF : array
        Contains the coefficients (same row-column formatting as argument 
        `WF`) of the expansion of wave-functions corresponding to 
        plane-waves of energy smaller than `Ecut`. Columns (plane-waves) 
        are shorted based on their energy, from smaller to larger.
    igall : array
        Every row corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of columns is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth column stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) column contains the index of the first (last) plane-wave with 
        the same energy as the plane-wave of the current column.
    """
    thresh = 1e-4  # default thresh to distinguish energies of plane-waves
    KG = (kg + kpt).dot(RecLattice)
    eKG = Hartree_eV * bohr_angstrom**2 * (la.norm(KG, axis=1) ** 2) / 2
    log_message(
        f"Found cutoff: {Ecut0:12.6f} eV   Largest plane wave energy in K-point {ik:4d}: {np.max(eKG):12.6f} eV",
        verbosity=verbosity, level=2
    )
    assert Ecut0 * 1.000000001 > np.max(eKG)
    sel = np.where(eKG < Ecut)[0]

    KG = KG[sel]
    kg = kg[sel]
    eKG = eKG[sel]
    srt = np.argsort(eKG)
    eKG = eKG[srt]
    igall = np.zeros((len(sel), 6), dtype=int)
    igall[:, :3] = kg[srt]
    igall[:, 3] = srt
    wall = (
        [0] + list(np.where(eKG[1:] - eKG[:-1] > thresh)[0] + 1) + [igall.shape[0]]
    )
    for i in range(len(wall) - 1):
        igall[wall[i]: wall[i + 1], 4] = wall[i]
        igall[wall[i]: wall[i + 1], 5] = wall[i + 1]


    WF = WF[:, sel[srt], :]

    return WF, igall, eKG


def transform_gk(kpt, ig, A, kpt_other=None):
    B = np.linalg.inv(A)
    # kptTr = B.dot(kpt)
    kptTr = kpt @ B  # kptTr is a direct coordinate of the transformed k-point
    if kpt_other is None:
        kpt_other = kptTr
    dkpt = np.array(np.round(kptTr - kpt_other), dtype=int)

    if not np.isclose(dkpt, kptTr - kpt_other).all():
        raise NotSymmetryError(
            f"The k-point {kpt} is transformed point {kptTr}  that is non-equivalent to the final point {kpt_other} "
            f"under transformation\n {A}"
        )
    igTr = ig[:, :3] @ B + dkpt[None, :]
    igTr = np.array(np.round(igTr), dtype=int)
    return kpt_other, igTr



def transformed_g_order(kpt, ig, A, kpt_other=None, ig_other=None, inverse=False):
    """
    Determines how the transformation matrix `A` reorders the reciprocal
    lattice vectors taking part in the plane-wave expansion of wave-functions.

    Parameters
    ----------
    kpt : array, shape=(3,)
        Direct coordinates of the k-point.
    ig : array
        Every column corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of rows is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth row stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) row contains the index of the first (last) groups of 
        plane-waves of identical energy.
    ig_other : array, default=None
        If `ig` is not the same as `ig_other` the order of the rotates g-vectors 
        is determined by `ig_other`. (for transformations between different k-points)
    A : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.

    Returns
    -------
    rotind : array
        `rotind[i] = j` if `B @ ig[i] == ig_other[j]`. if inverse is `False`,
        'rotind[j] = i' if `B @ ig[i] == ig_other[j]`. if inverse is `True`.

        where `B = np.linalg.inv(A).T`
    """
    assert (ig_other is None) == (kpt_other is None), "ig_other and kpt_other must be provided (or not) together"
    if ig_other is None:
        ig_other = ig
        kpt_other = kpt
    _, igTr = transform_gk(kpt, ig, A, kpt_other)
    ng = ig.shape[0]
    rotind = -np.ones(ng, dtype=int)
    for i in range(ng):
        for j in range(ig[i, 4], ig[i, 5]):
            if (igTr[i, :] == ig_other[j, :3]).all():
                if inverse:
                    rotind[j] = i
                else:
                    rotind[i] = j
                break

    for i in range(ng):
        if rotind[i] == -1:

            raise RuntimeError(
                f"Error in the transformation of plane-waves in k-point={kpt}: "
                f"No pair found for the g-vector igTr[{i}]={igTr[i]} "
                f"obtained when transforming the g-vector ig[{i}]={ig_other[i, :3]} "
                f"with the matrix  B=inv(A).T with A={A}"
            )
    return rotind


def symm_eigenvalues(
    K, WF, igall, A, S, T, spinor, block_ind=None
):
    """
    Calculate the traces of a symmetry operation for the wave-functions in a 
    particular k-point.

    Parameters
    ----------
    K : array, shape=(3,)
        Direct coordinates of the k-point.
    WF : array
        `WF[i,j, s]` contains the coefficient corresponding to :math:`j^{th}`
        plane-wave in the expansion of the wave-function in :math:`i^{th}`
        band. and s^{th} spin component
        It contains only plane-waves of energy smaller than `Ecut`.
    igall : array
        Returned by `__sortIG`.
        Every row corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of columns is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth column stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) column contains the index of the first (last) plane-wave with 
        the same energy as the plane-wave of the current column.
    A : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    S : array, shape=(2,2)
        Matrix describing how spinors transform under the symmetry.
    T : array, shape=(3,)
        Translational part of the symmetry operation, in terms of the basis 
        vectors of the unit cell.
    spinor : bool
        `True` if wave-functions are spinors, `False` if they are scalars.

    Returns
    -------
    array
        Each element is the trace of the symmetry operation in a wave-function.
    """
    if block_ind is not None:
        return symm_eigenvalues_blocks(K, WF, igall, A, S, T, spinor, block_ind)
    multZ = np.exp(
        -1.0j * (2 * np.pi * (igall[:, :3] + K[None, :])  @ (np.linalg.inv(A) @ T))
    )
    igrot = transformed_g_order(kpt=K, ig=igall, A=A)
    if spinor:
        return cached_einsum('igs,igt,st->ig', WF[:, igrot].conj(), WF[:, :], S).dot(multZ)
    else:
        return (WF[:, igrot, 0].conj() * WF[:, :, 0]).dot(multZ)


def symm_eigenvalues_blocks(K, WF, igall, A, S, T, spinor, block_ind):
    """	
    same as symm_eigenvalues, but uses symm_matrix to calculate the traces	
    """
    matrix_blocks = symm_matrix(K, WF, igall, A, S, T, spinor, return_blocks=True, block_ind=block_ind)
    traces = []
    for block in matrix_blocks:
        n = block.shape[0]
        traces += [np.trace(block) / n] * n
    return np.array(traces)





def symm_matrix(
    K, WF, igall, A, S, T, spinor,
    time_reversal=False,
    WF_other=None, igall_other=None, K_other=None,
    block_ind=None,
    return_blocks=False,
    ortogonalize=True,
    unitary=True,
    unitary_params={},
    Ecut=None,
    eKG=None  # Ecut is not used in this function, but it is needed for compatibility
):
    """
    Computes the matrix S_mn such that
    {A|T} |Psi_nk> = sum_m |Psi_mk'> * S_mn

    WARNING : In the versions from 1.10.0 to 2.2.0 this function was giving the transposed matrices.

    Parameters
    ----------
    K : array, shape=(3,)
        Direct coordinates of the k-point. 
    WF : array
        `WF[i,j,s]` contains the coefficient corresponding to :math:`j^{th}`
        plane-wave in the expansion of the wave-function in :math:`i^{th}`
        band. and s^{th} spin component.
        It contains only plane-waves if energy smaller than `Ecut`.
    igall : array
        Returned by `__sortIG`.
        Every row corresponds to a plane-wave of energy smaller than 
        `Ecut`. The number of columns is 6: the first 3 contain direct 
        coordinates of the plane-wave, the fourth column stores indices needed
        to short plane-waves based on energy (ascending order). Fitfth 
        (sixth) column contains the index of the first (last) plane-wave with 
        the same energy as the plane-wave of the current column.
    WF_other, igall_other : array, default=None
        if provided, transformation to a different point is calculated.
    A : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    S : array, shape=(2,2)
        Matrix describing how spinors transform under the symmetry.
    T : array, shape=(3,)
        Translational part of the symmetry operation, in terms of the basis 
        vectors of the unit cell.
    time_reversal : bool, default=False
        If `True`, the time-reversal symmetry is applied.
    spinor : bool
        `True` if wave functions are spinors, `False` if they are scalars.
    block_ind : list( tuple(int,int) ), default=None
        If provided, only the diagonal blocks specified in the list are computed 
        The list contains tuples of the form (m,n) where m and n are the indices
        of the blocks to be computed.. i.e. S[m:n,m:n] is computed.
    return_blocks : bool, default=False
        If `True`, returns the diagonal blocks as a list. Otherwise, returns the
        matrix composed of those blocks.
    ortogonalize : bool, default=True
        If `True`, the matrix is orthogonalized. Set to `False` for speedup. 
        (in general it is not needed, but just in case)
    unitary : bool, default=True
        If `True`, the matrix is orthogonalized (made unitary). Set to `False` for speedup. 
        (in general it is not needed, but just in case)
    Returns
    -------
    array
        Matrix of the symmetry operation in the basis of eigenstates of the 
        Bloch Hamiltonian :math:`H(k)`.
    """
    assert (WF_other is None) == (igall_other is None) == (K_other is None), "WF_other and igall_other must be provided (or not) together"
    if Ecut is not None:
        assert eKG is not None, "Ecut is provided, but eKG is not"
        select = np.where(eKG <= Ecut)[0]
        npw_cut = igall[select.max(), 5]
        igall = igall[:npw_cut]
        WF = WF[:, :npw_cut, :]
        if WF_other is not None:
            igall_other = igall_other[:npw_cut]
            WF_other = WF_other[:, :npw_cut, :]
        return symm_matrix(
            K=K, WF=WF, igall=igall, A=A, S=S, T=T,
            spinor=spinor, time_reversal=time_reversal,
            WF_other=WF_other, igall_other=igall_other, K_other=K_other,
            block_ind=block_ind, return_blocks=return_blocks,
            ortogonalize=ortogonalize, unitary=unitary,
            unitary_params=unitary_params
        )

    if WF_other is None:
        WF_other = WF
        igall_other = igall
        K_other = K
    if block_ind is None:
        block_ind = np.array([(0, WF.shape[0])])

    unitary_params_loc = {
        "warning_threshold": 1e-3,
        "error_threshold": 1e-2,
        "check_upper": False,
        "warn_upper": False,
        "nbands_upper_skip": 2
    }
    for key, val in unitary_params.items():
        if key not in unitary_params_loc:
            warnings.warn(f"You provided a parameter {key}:{val} for unitarity check in symm_matrix, which is not recognized. Probably this is a typo."
                          f"the recognized parameters and theitr default values are {unitary_params_loc}")
    unitary_params_loc.update(unitary_params)
    multZ = np.exp(-2j * np.pi * (igall_other[:, :3] + K_other[None, :]) @ T)

    if time_reversal:
        A = -A
        WF = WF.conj()
        # multZ = multZ.conj() # this is not needed because igall_other and K_other are already reversed (because A=-A)
        if spinor:
            S = np.array([[0, 1], [-1, 0]]) @ S.conj()

    igrot = transformed_g_order(kpt=K, ig=igall, A=A, ig_other=igall_other, kpt_other=K_other, inverse=True)
    WFrot = WF[:, igrot, :] * multZ[None, :, None]
    if spinor:
        WFrot = cached_einsum("ts,mgs->mgt", S, WFrot)
    WFrot = np.hstack([WFrot[:, :, s] for s in range(WFrot.shape[2])])
    WF_other = np.hstack([WF_other[:, :, s] for s in range(WF_other.shape[2])])
    # WFrot = WFrot.reshape((WFrot.shape[0], -1), order='F')
    block_list = []
    NB = WF.shape[0]
    for b1, b2 in block_ind:
        WFinv = right_inverse(WF_other[b1:b2])
        block = np.dot(WFrot[b1:b2, :], WFinv).T
        if unitary:
            if not unitary_params_loc["check_upper"] and b2 >= NB - unitary_params_loc["nbands_upper_skip"]:
                error_threshold = 100
            else:
                error_threshold = unitary_params_loc["error_threshold"]
            if not unitary_params_loc["warn_upper"] and b2 >= NB - unitary_params_loc["nbands_upper_skip"]:
                warning_threshold = 100
            else:
                warning_threshold = unitary_params_loc["warning_threshold"]
            block = orthogonalize(block,
                                  warning_threshold=warning_threshold,
                                  error_threshold=error_threshold,
                                  debug_msg=f"symm_matrix: block {b1}:{b2} of {WF.shape[0]}")
        block_list.append(block)
    if return_blocks:
        return block_list
    else:
        nwfout = sum(b2 - b1 for b1, b2 in block_ind)
        M = np.zeros((nwfout, nwfout), dtype=complex)
        i = 0
        for (b1, b2), block in zip(block_ind, block_list):
            b = b2 - b1
            M[i:i + b, i:i + b] = block
            i += b
        return M


def right_inverse(A):
    """
    Compute the right inverse of a rectangular matrix A (m x n) where m < n.

    Parameters:
    A (numpy.ndarray): The input matrix of shape (m, n).

    Returns:
    numpy.ndarray: The right inverse of A of shape (n, m).
    """
    assert A.shape[0] <= A.shape[1], "Matrix must be rectangular, and m <= n."
    A_T = A.T.conj()
    return A_T @ np.linalg.inv(A @ A_T)
