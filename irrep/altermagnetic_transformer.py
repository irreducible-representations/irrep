import numpy as np

from .spacegroup import SpaceGroup
from .utility import all_close_mod1
from .symmetry_operation import SymmetryOperation, get_atom_map


class AltermagneticTransformer:

    def __init__(self, calculator,
                 symmetrizer_up,
                 alter_symop=None,
                 rotation_latt=None,
                 translation_latt=None,
                ):
        spacegroup_up = symmetrizer_up.spacegroup
        if alter_symop is None:
            alter_symop = SymmetryOperation(rot=rotation_latt,
                                      trans=translation_latt,
                                      Lattice=spacegroup_up.Lattice,
                                      spinor=False
                                      )
        self.alter_symop = alter_symop
        self.alter_symop.set_gpaw(calculator)
        # self.symops = [symop * alter_symop for symop in spacegroup_up.symmetries]
        self.alter_map = []  # alter_map[i] = j means ki = alter_map_symops[i] * kj
        self.alter_map_isym = []
        self.symmetry_operations = []
        self.k_intermediate = []
        for kirr in symmetrizer_up.kptirr:
            kpt_red = symmetrizer_up.kpoints_all[kirr]
            kpt_red_before_transform = alter_symop.transform_k(kpt_red, inverse=True)
            for ik, kpt in enumerate(symmetrizer_up.kpoints_all):
                if all_close_mod1(kpt, kpt_red_before_transform):
                    self.alter_map.append(symmetrizer_up.kpt2kptirr[ik])
                    isym = symmetrizer_up.kpt2kptirr_sym[ik]
                    self.alter_map_isym.append(isym)
                    self.symmetry_operations.append(symmetrizer_up.spacegroup.symmetries[isym])
                    self.k_intermediate.append(kpt)
                    break
            else:
                raise RuntimeError(f"No corresponding k-point found for R({kpt_red})={kpt_red_before_transform} out of \n{symmetrizer_up.kpoints_all} under the altermagnetic symmetry operation.")

    @classmethod
    def from_gpaw(cls, calculator, symmetrizer_up, nskip_symmetries=0):
        symop = find_altermagnetic_symmetry(lattice=calculator.atoms.cell,
                                            positions=calculator.atoms.get_scaled_positions(),
                                            typat=calculator.atoms.get_atomic_numbers(),
                                            magmom=calculator.get_magnetic_moments(),
                                            nskip=nskip_symmetries)
        return cls(calculator, symmetrizer_up, rotation_latt=symop.rotation, translation_latt=symop.translation)

    def get_kpoint_down(self, ikirr, kpoints_up):
        ik_origin = self.alter_map[ikirr]
        KPtransformed = kpoints_up[ik_origin].get_transformed_copy(
            symmetry_operation=self.symmetry_operations[ikirr],
            k_new=self.k_intermediate[ikirr])
        KPtransformed = KPtransformed.get_transformed_copy(
            symmetry_operation=self.alter_symop,
            k_new=kpoints_up[ikirr].k)
        # TODO : unify the transformations, using just their product
        return KPtransformed


def find_altermagnetic_symmetry(lattice, positions, typat, magmom, nskip=0):
    magmom = np.array(magmom)
    assert magmom.ndim == 1, "Magnetic moments must be a 1D array."
    positions = np.array(positions)
    assert positions.ndim == 2 and positions.shape[1] == 3, "Positions must be a 2D array with shape (N, 3)."
    typat = np.array(typat)
    assert typat.ndim == 1, "Atom types must be a 1D array."
    assert len(positions) == len(typat)
    assert len(positions) == len(magmom), "Number of positions, atom types, and magnetic moments must match."

    spacegroup_nomag = SpaceGroup.from_cell(real_lattice=lattice, positions=positions, typat=typat, include_TR=False)
    if nskip >= len(spacegroup_nomag.symmetries) // 2:
        raise ValueError(f"nskip={nskip} is too large for the number of symmetries={len(spacegroup_nomag.symmetries)}.")
    cnt = 0
    for symop in spacegroup_nomag.symmetries:
        atommap, T = get_atom_map(symop, positions)
        # print (f"Checking symmetry operation: {symop}, {atommap=}, {T=}")
        assert np.all(typat[atommap] == typat), "Atom types must match under symmetry operation."
        magmom_transformed = magmom[atommap]
        print(f"{magmom_transformed=}, {atommap=},")
        if np.allclose(magmom_transformed, -magmom):
            cnt += 1
            if cnt > nskip:
                return symop
    else:
        raise ValueError("No altermagnetic symmetry operation found for the given magnetic moments.")
