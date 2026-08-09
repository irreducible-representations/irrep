import numpy as np

from .spacegroup import SpaceGroup
from .utility import all_close_mod1
from .symmetry_operation import SymmetryOperation, get_atom_map


class AltermagneticTransformer:

    def __init__(self, calculator,
                 spacegroup_up,
                 alter_symop=None,
                 rotation_latt=None,
                 translation_latt=None
                ):

        if alter_symop is None:
            alter_symop = SymmetryOperation(rot=rotation_latt,
                                      trans=translation_latt,
                                      Lattice=spacegroup_up.Lattice,
                                      spinor=False
                                      )
        self.alter_symop = alter_symop
        self.symops = [symop * alter_symop for symop in spacegroup_up.symmetries]
        ibz_kpoints = calculator.get_ibz_k_points()
        nq = len(ibz_kpoints)
        self.alter_map = -np.ones(nq, dtype=int)  # alter_map[i] = j means ki = alter_map_symops[i] * kj
        alter_map_symops = -np.ones(nq, dtype=int)
        for ibz_index_up, k_up in enumerate(ibz_kpoints):
            for isym, symop in enumerate(self.symops):
                k_new = symop.transform_k(k_up)
                # print (f"{ibz_index_up=} {k_up=} {k_new=} {isym=}")
                for ibz_index_down, k_down in enumerate(ibz_kpoints):
                    if self.alter_map[ibz_index_down] != -1:
                        continue
                    if all_close_mod1(k_new, k_down):
                        self.alter_map[ibz_index_down] = ibz_index_up
                        alter_map_symops[ibz_index_down] = isym
                        break
        if np.any(self.alter_map == -1):
            raise ValueError("Not all k-points in the IBZ have a corresponding k-point under the altermagnetic symmetry operation.\n"
                             f"Missing k-points:\n {ibz_kpoints[self.alter_map == -1]}"
                             )
        for symop in self.symops:
            symop.set_gpaw(calculator)

    @classmethod
    def from_gpaw(cls, calculator):
        magmom = calculator.get_magnetic_moments()
        spacegroup_up = SpaceGroup.from_gpaw(calculator, include_TR=False, magmom=magmom * 0)
        symop = find_altermagnetic_symmetry(lattice=spacegroup_up.Lattice,
                                            positions=spacegroup_up.positions,
                                            typat=spacegroup_up.typat,
                                            magmom=magmom)
        return cls(calculator, spacegroup_up, rotation_latt=symop.rotation, translation_latt=symop.translation)



def find_altermagnetic_symmetry(lattice, positions, typat, magmom):
    magmom = np.array(magmom)
    assert magmom.ndim == 1, "Magnetic moments must be a 1D array."
    positions = np.array(positions)
    assert positions.ndim == 2 and positions.shape[1] == 3, "Positions must be a 2D array with shape (N, 3)."
    typat = np.array(typat)
    assert typat.ndim == 1, "Atom types must be a 1D array."
    assert len(positions) == len(typat)
    assert len(positions) == len(magmom), "Number of positions, atom types, and magnetic moments must match."

    spacegroup_nomag = SpaceGroup.from_cell(real_lattice=lattice, positions=positions, typat=typat, include_TR=False)
    for symop in spacegroup_nomag.symmetries:
        atommap, T = get_atom_map(symop, positions)
        # print (f"Checking symmetry operation: {symop}, {atommap=}, {T=}")
        assert np.all(typat[atommap] == typat), "Atom types must match under symmetry operation."
        magmom_transformed = magmom[atommap]
        # print (f"{magmom_transformed=}")
        if np.allclose(magmom_transformed, -magmom):
            return symop
    else:
        raise ValueError("No altermagnetic symmetry operation found for the given magnetic moments.")
