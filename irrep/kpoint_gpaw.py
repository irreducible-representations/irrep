import numpy as np
from .kpoint import KpointAbstract


class KpointGPAW(KpointAbstract):
    """ a container to store wavefunctions and eigenvalues at a k-point from GPAW.

    Args:
        kpt (array): K-point coordinates in Cartesian coordinates.
        weight (float): Weight of the k-point.
        ibzkpt (int): Index of the k-point in the irreducible Brillouin zone.
        nkpts (int): Total number of k-points.
        nspden (int): Spin density index.
        nspwfc (int): Spin wavefunction index.
        gpts (int): Number of G-vectors.
        occ (float): Occupation number.
        eigenvalues (array): Eigenvalues at the k-point.
        wavefunctions (array): Wavefunctions at the k-point.
    """

    def __init__(self, kpt, wavefunction=None, proj=None, nbands=None,
                 RecLattice=None,
                 atom_positions=None):
        super().__init__(kpt=kpt, num_bands=nbands)
        self.wavefunction = wavefunction
        self.proj = proj
        self.nbands = nbands
        self.RecLattice = RecLattice
        self.atom_positions = atom_positions

    @classmethod
    def from_gpaw(self, calc, ibz_index, ispin, RecLattice=None):
        kpt = calc.wfs.kpt_qs[ibz_index][ispin]
        k = calc.get_ibz_k_points()[ibz_index]
        nbands = calc.wfs.bd.nbands
        # Get projections in ibz
        proj = kpt.projections.new(nbands=nbands, bcomm=None)
        proj.array[:] = kpt.projections.array[:nbands]
        if RecLattice is None:
            RecLattice = calc.wfs.gd.reciprocal_lattice
        atom_positions = calc.atoms.get_positions()

        wavefunction = np.array([calc.wfs.get_wave_function_array(n, ibz_index, ispin, periodic=True)
                                 for n in range(nbands)])
        return KpointGPAW(kpt=k, wavefunction=wavefunction, proj=proj, nbands=nbands, RecLattice=RecLattice, atom_positions=atom_positions)


    def get_transformed_copy(self, symmetry_operation, k_new):
        """Get a transformed copy of the KpointGPAW object according to the given symmetry operation.

        Args:
            symmetry_operation (SymmetryOperation): Symmetry operation to apply.
            k_new (array): Target k-point in fractional coordinates.

        Returns:
            KpointGPAW: Transformed copy of the KpointGPAW object.
        """
        phase = np.exp(-2j * np.pi * symmetry_operation.transform_k(self.k)  @ symmetry_operation.translation)
        new_wavefunction = rotate_pseudo_wavefunction(self.wavefunction, symmetry_operation, self.k, k_new) * phase
        new_proj = rotate_projection(self.proj, symmetry_operation, self.k, k_new, phase=1) 
        return KpointGPAW(kpt=k_new, wavefunction=new_wavefunction, proj=new_proj, nbands=self.nbands)


def rotate_projection(projections, symop, k_origin, k_target, phase=1.0):
    """
    Rotate the projection coefficients according to the given symmetry operation

    Parameters
    ----------
    proj : Projections
        the projection coefficients
    symop : irrep.SymmetryOperation
        the symmetry operation
    k_origin : np.ndarray(shape=(3,), dtype=float)
        the original k-point in the basis of the reciprocal lattice
    k_target : np.ndarray(shape=(3,), dtype=float)
        the target k-point in the basis of the reciprocal lattice

    Returns
    -------
    proj_rot : Projections
        the rotated projection coefficients
    """
    mapped_projections = projections.new()

    U_aii = symop.get_U_aii_gpaw(kpoint=k_target)
    for a, (b, U_ii) in enumerate(zip(symop.atom_map, U_aii)):
        # Map projections
        Pin_ni = projections[b]
        Pout_ni = Pin_ni @ U_ii
        if symop.time_reversal:
            Pout_ni = np.conj(Pout_ni)
        # Store output projections
        I1, I2 = mapped_projections.map[a]
        mapped_projections.array[..., I1:I2] = Pout_ni*phase
    return mapped_projections


def rotate_pseudo_wavefunction(psi_n_grid, symop, k_origin, k_target):
    """
    Rotate the pseudo wavefunction according to the given symmetry operation

    Parameters
    ----------
    psi_nG : np.ndarray(shape=(NB, n1, n2, n3), dtype=complex)
        the pseudo wavefunction in G-space
    symop : irrep.SymmetryOperation
        the symmetry operation
    k_origin : np.ndarray(shape=(3,), dtype=float)
        the original k-point in the basis of the reciprocal lattice
    k_target : np.ndarray(shape=(3,), dtype=float)
        the target k-point in the basis of the reciprocal lattice

    Returns
    -------
    psi_nG_rot : np.ndarray(shape=(NB, NG), dtype=complex)
        the rotated pseudo wavefunction in G-space
    """
    Nc = psi_n_grid.shape[1:]
    # Nc_tot = np.prod(Nc)
    if not symop.is_identity:
        # NB = psi_n_grid.shape[0]

        # # First way
        # # indx_inv = symop.transform_grid_indices(Nc, inverse=True)
        # # psi_n_grid = psi_n_grid.reshape(NB, -1)[:, indx_inv].reshape((NB,) + Nc)

        # # Second way (slightly slower)
        # indx = symop.transform_grid_indices(Nc, inverse=False)
        # psi_n_grid = psi_n_grid.copy()
        # psi_n_grid = psi_n_grid.reshape(NB, Nc_tot)
        # psi_n_grid[:, indx] = psi_n_grid
        # psi_n_grid = psi_n_grid.reshape((NB,) + Nc)
        psi_n_grid = symop.transform_grid_data(psi_n_grid)


    if symop.time_reversal:
        psi_n_grid = np.conj(psi_n_grid)
    kpt_shift = k_target - symop.transform_k(k_origin)
    kpt_shift_int = np.round(kpt_shift).astype(int)
    assert np.allclose(kpt_shift, kpt_shift_int), f"k-point shift {kpt_shift} is not a reciprocal lattice vector"
    for i, ksh in enumerate(kpt_shift_int):
        if ksh != 0:
            phase = np.exp(-2j * np.pi * ksh * np.arange(Nc[i]) / Nc[i]).reshape((1,) * (i + 1) + (Nc[i],) + (1,) * (2 - i))
            psi_n_grid = psi_n_grid * phase
    return psi_n_grid


# def U_aii(self, R_aii, spos_ac, atom_map):
#     """Phase corrected rotation matrices for the PAW projections."""
#     U_aii = []
#     for a, R_ii in enumerate(R_aii):
#         # The symmetry transformation maps atom "a" to a position which is
#         # related to atom "b" by a lattice vector (but which does not
#         # necessarily lie within the unit cell)
#         b = atom_map[a]
#         cell_shift_c = spos_ac[a] @ self.U_cc - spos_ac[b]  # add translation here ?
#         assert np.allclose(cell_shift_c.round(), cell_shift_c, atol=1e-6)
#         # This means, that when we want to extract the projections at K for
#         # atom a according to psi_K(r_a) = psi_ik(U^T r_a), we need the
#         # projections at U^T r_a for k-point ik. Since we only have the
#         # projections within the unit cell we need to multiply them with a
#         # phase factor according to the cell shift.
#         phase_factor = np.exp(2j * np.pi * self.ik_c @ cell_shift_c)
#         U_ii = R_ii.T * phase_factor
#         U_aii.append(U_ii)

#     return U_aii
