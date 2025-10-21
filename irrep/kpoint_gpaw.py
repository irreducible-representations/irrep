import numpy as np
from .kpoint import KpointAbstract
from .utility import cached_einsum


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
        atom_positions = calc.atoms.get_scaled_positions()

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
        new_wavefunction = symmetry_operation.rotate_pseudo_wavefunction(psi_n_grid=self.wavefunction, k_origin=self.k, k_target=k_new)
        new_proj = symmetry_operation.rotate_projection(self.proj, self.k, k_new)
        return KpointGPAW(kpt=k_new, wavefunction=new_wavefunction, proj=new_proj, nbands=self.nbands,
                          RecLattice=self.RecLattice, atom_positions=self.atom_positions)


class OverlapPAW:

    def __init__(self, wfs):
        self.dO_aii = {}
        for a in wfs.kpt_u[0].projections.map:
            self.dO_aii[a] = wfs.setups[a].dO_ii
        self.dv = wfs.gd.dv

    def product(self, KP1, KP2,
                include_paw=True,
                include_pseudo=True,
                bk=None,
                ):
        wf1 = KP1.wavefunction
        proj1 = KP1.proj
        wf2 = KP2.wavefunction
        proj2 = KP2.proj
        assert wf1.ndim == 4
        assert wf2.ndim == 4
        assert wf1.shape[1:] == wf2.shape[1:], f"wavefunction grids do not match: {wf1.shape[1:]} vs {wf2.shape[1:]}"
        prod = np.zeros((wf1.shape[0], wf2.shape[0]), dtype=complex)
        if include_pseudo:
            prod += cached_einsum('aijk,bijk->ab', wf1.conj(), wf2) * self.dv
        if include_paw:
            positions = KP1.atom_positions
            na = len(positions)
            if bk is None:
                phases = np.ones(na, dtype=complex)
            else:
                phases = np.exp(-2j * np.pi * (positions @ bk))
            for a, dO_ii in self.dO_aii.items():
                prod += (proj1[a].conj() @ dO_ii @ (proj2[a].T)) * phases[a]
        return prod
