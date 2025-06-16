
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
##  e-mail: stepan.tsirkin@epfl.ch                               #
##################################################################


import numpy as np
from irreptables import IrrepTable
from scipy.optimize import minimize

from .spacegroup import SpaceGroup

from .utility import log_message


class SpaceGroupIrreps(SpaceGroup):
    """
    This class is for internal usage of irrep. While the parent class is for wider use (e.g. in wannierberri)

    has additional attributes:

    Attributes
    ----------
    symmetries_tables : list
        Attribute `symmetries` of class `IrrepTable`. Each component is an 
        instance of class `SymopTable` corresponding to a symmetry operation
        in the "point-group" of the space-group.
    u_symmetries_tables : list
        Attribute `u_symmetries` of class `IrrepTable`. Each component is an 
        instance of class `SymopTable
        corresponding to a unitary symmetry operation in the "point-group"
        of the space-group.
    au_symmetries_tables : list
        Attribute `au_symmetries` of class `IrrepTable`. Each component is an 
        instance of class `SymopTable
        corresponding to an antiunitary symmetry operation in the "point-group"
        of the space-group.


    """

    def set_irreptables(self,
            refUC=None,
            shiftUC=None,
            search_cell=False,
            trans_thresh=1e-5,
            no_match_symmetries=False,
            verbosity=0):
        # Load symmetries from the space group's table
        irreptable = IrrepTable(self.number_str, self.spinor, magnetic=self.magnetic, v=verbosity)
        self.u_symmetries_tables = irreptable.u_symmetries
        self.au_symmetries_tables = irreptable.au_symmetries

        if not search_cell:
            no_match_symmetries = True
        else:
            no_match_symmetries = False

        # Determine refUC and shiftUC according to entries in CLI
        self.refUC, self.shiftUC = self.determine_basis_transf(
            refUC_cli=refUC,
            shiftUC_cli=shiftUC,
            refUC_lib=self.refUC,
            shiftUC_lib=self.shiftUC,
            search_cell=search_cell,
            trans_thresh=trans_thresh,
            verbosity=verbosity
        )


        # Check matching of symmetries in refUC. If user set transf.
        # in the CLI and symmetries don't match, raise a warning.
        # Otherwise, if transf. was calculated automatically,
        # matching of symmetries was checked in determine_basis_transf
        if no_match_symmetries:
            self.spin_transf = np.eye(2)  # for printing
        else:
            sorted_symmetries = []
            try:
                ind, dt, signs, U = self.match_symmetries(signs=self.spinor,
                                                          trans_thresh=trans_thresh,
                                                          only_u_symmetries=False)
                args = np.argsort(ind)
                self.spin_transf = U
                symmetries = self.symmetries
                for i, i_ind in enumerate(args):
                    symmetries[i_ind].ind = i + 1
                    symmetries[i_ind].sign = signs[i_ind]
                    sorted_symmetries.append(symmetries[i_ind])
            except RuntimeError:
                if search_cell:  # symmetries must match to identify irreps
                    raise RuntimeError(
                        "refUC and shiftUC don't transform the cell to one where "
                        "symmetries are identical to those read from tables. "
                        "Try without specifying refUC and shiftUC.")
                elif refUC is not None or shiftUC is not None:
                    # User specified refUC or shiftUC in CLI. He/She may
                    # want the traces in a cell that is not neither the
                    # one in tables nor the DFT one
                    log_message("WARNING: refUC and shiftUC don't transform the cell to "
                                "one where symmetries are identical to those read from "
                                "tables. If you want to achieve the same cell as in "
                                "tables, try not specifying refUC and shiftUC.",
                                verbosity, 1)
            self.symmetries = sorted_symmetries
        self.irreps_are_set = True

    def check_irreps_set(self):
        assert self.irreps_are_set, "Cannot proceed before `SpaceGroupIrreps.set_irreptables()` is called. " \


    def show(self, symmetries=None):
        """
        Print description of space-group and symmetry operations.

        Parameters
        ----------
        symmetries : int, default=None
            Index of symmetry operations whose description will be printed. 
            Run `IrRep` with flag `onlysym` to check the index corresponding 
            to each symmetry operation.
        """

        print('')
        print("\n ---------- CRYSTAL STRUCTURE ---------- \n")
        print('')

        # Print cell vectors in DFT and reference cells
        vecs_refUC = np.dot(self.real_lattice, self.refUC).T
        print('Cell vectors in angstroms:\n')
        print(f"{'Vectors of DFT cell':^32}|{'Vectors of REF. cell':^32}")
        for i in range(3):
            vec1 = self.real_lattice[i]
            vec2 = vecs_refUC[i]
            s = f"a{i:1d} = {vec1[0]:7.4f}  {vec1[1]:7.4f}  {vec1[2]:7.4f}  "
            s += "|  "
            s += f"a{i:1d} = {vec2[0]:7.4f}  {vec2[1]:7.4f}  {vec2[2]:7.4f}"
            print(s)
        print()

        # Print atomic positions
        print('Atomic positions in direct coordinates:\n')
        print(f"{'Atom type':^} | {'Position in DFT cell':^25} | {'Position in REF cell':^25}")
        positions_refUC = np.linalg.inv(self.refUC) @ np.transpose(self.positions - self.shiftUC)
        positions_refUC = positions_refUC.T % 1.0
        for itype, pos1, pos2 in zip(self.typat, self.positions, positions_refUC):
            s = f'{itype:^9d} | '
            s += '  '.join(f'{x:7.4f}' for x in pos1) + ' | '
            s += '  '.join(f'{x:7.4f}' for x in pos2)
            print(s)

        print()
        print('\n ---------- SPACE GROUP ----------- \n')
        print()
        print(f"Space group: {self.name} (# {self.number_str})")
        print(f"Number of unitary symmetries: {len(self.u_symmetries)} (mod. lattice translations)")
        if self.magnetic:
            print(f"Number of antiunitary symmetries: {len(self.au_symmetries)} (mod. lattice translations)")
        refUC_print = self.refUC.T  # print following convention in paper
        print(
            "\nThe transformation from the DFT cell to the reference cell of tables is given by:\n"
            f"        | {''.join(f'{el:8.4f}' for el in refUC_print[0])} |\n"
            f"refUC = | {''.join(f'{el:8.4f}' for el in refUC_print[1])} |    shiftUC = {np.round(self.shiftUC, 5)}\n"
            f"        | {''.join(f'{el:8.4f}' for el in refUC_print[2])} |\n"
        )

        for symop in self.symmetries:
            if symmetries is None or symop.ind in symmetries:
                symop.show(refUC=self.refUC, shiftUC=self.shiftUC, U=self.spin_transf)


    def json(self, symmetries=None):
        '''
        Prepare dictionary with info of space group to save in JSON

        Returns
        -------
        d : dict
            Dictionary with info about space group
        '''

        d = {}

        if (np.allclose(self.refUC, np.eye(3)) and
                np.allclose(self.shiftUC, np.zeros(3))):
            cells_match = True
        else:
            cells_match = False

        d = {"name": self.name,
             "number": self.number_str,
             "spinor": self.spinor,
             "num symmetries": self.size,
             "cells match": cells_match,
             "symmetries": {},
             "magnetic": self.magnetic
             }

        for sym in self.symmetries:
            if symmetries is None or sym.ind in symmetries:
                d["symmetries"][sym.ind] = sym.json_dict(self.refUC, self.shiftUC)

        return d


    def write_trace(self):
        """
        Construct description of matrices of symmetry operations of the 
        space-group in the format: 
        {{R|t}}-> R11,R12,...,R23,R33,t1,t2,t3 and, when SOC was included, the 
        elements of the matrix describing the transformation of the spinor in 
        the format:
        Re(S11),Im(S11),Re(S12),...,Re(S22),Im(S22).

        Returns
        -------
        str
            String describing matrices of symmetry operations.
        """

        res = f" {self.size}\n"
        # In the following lines, one symmetry operation for each operation of the point group n"""
        for symop in self.symmetries:
            res += symop.str2(refUC=self.refUC, shiftUC=self.shiftUC, write_tr=self.magnetic)
        return (res)

    def str(self):
        """
        Create a string to describe of space-group and its symmetry operations.

        Returns
        -------
        str
            Description to print.
        """
        return (
            f"SG={self.number_str}\n"
            f"name={self.name}\n"
            f"nsym={self.size}\n"
            f"spinor={self.spinor}\n"
            "symmetries=\n" +
            "\n".join(s.str(self.refUC, self.shiftUC) for s in self.symmetries) +
            "\n\n"
        )


    def get_irreps_from_table(self, kpname, K, verbosity=0):
        """
        Read irreps of the little-group of a maximal k-point. 

        Parameters
        ----------
        kpname : str
            Label of the maximal k-point.
        K : array, shape=(3,)
            Direct coordinates of the k-point.
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing

        Returns
        -------
        tab : dict
            Each key is the label of an irrep, each value another `dict`. Keys 
            of every secondary `dict` are indices of symmetries (starting from 
            1 and following order of operations in tables of BCS) and 
            values are traces of symmetries.

        Raises
        ------
        RuntimeError
            Translational or rotational parts read from tables and found by 
            `spglib` do not match for a symmetry operation.
        RuntimeError
            A symmetry from the tables matches with many symmetries found by 
            `spglib`.
        RuntimeError
            The label of a k-point given in the CLI matches with the label of a 
            k-point read from tables, but direct coordinates of these 
            k-vectors do not match.
        RuntimeError
            There is not any k-point in the tables whose label matches that 
            given in parameter `kpname`.
        """

        table = IrrepTable(self.number_str, self.spinor, magnetic=self.magnetic, v=verbosity)
        tab = {}
        for irr in table.irreps:
            if irr.kpname == kpname:
                k1 = np.round(np.linalg.inv(self.refUC.T).dot(irr.k), 5) % 1
                k2 = np.round(K, 5) % 1
                if not np.allclose(k1, k2):
                    raise RuntimeError(f"the kpoint {K} does not correspond to the point {kpname} "
                                       f"({np.round(irr.k, 3)} in refUC / {k1} in primUC) in the table")
                tab[irr.name] = {}
                for i, (sym1, sym2) in enumerate(zip(self.symmetries, table.symmetries)):
                    try:
                        dt = sym2.t - sym1.translation_refUC(self.refUC, self.shiftUC)
                        tab[irr.name][i + 1] = irr.characters[i + 1] * \
                            sym1.sign * np.exp(2j * np.pi * dt.dot(irr.k))
                    except KeyError:
                        pass
        if len(tab) == 0:
            raise RuntimeError(
                f"the k-point with name {kpname} is not found in the spacegroup {table.number_str}. found only :\n"
                "\n ".join("{kpname}({k}/{krefuc})".format(
                    kpname=irr.kpname,
                    k=irr.k,
                    krefuc=np.linalg.inv(self.refUC).dot(irr.k) % 1
                ) for irr in table.irreps)
            )
        return tab

    def determine_basis_transf(
            self,
            refUC_cli,
            shiftUC_cli,
            refUC_lib,
            shiftUC_lib,
            search_cell,
            trans_thresh,
            verbosity=0
    ):
        """ 
        Determine basis transformation to conventional cell. Priority
        is given to the transformation set by the user in CLI.

        Parameters
        ----------
        refUC_cli : array
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting. Set in CLI.
        shiftUC_cli : array
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting. Set in CLI.
        refUC_lib : array
            Obtained via spglib.
        shiftUC_lib : array
            Obtained via spglib. It may not be the shift taking the
            origin to the position adopted in the tables (BCS). For 
            example, origin choice 1 of ITA is adopted in spglib for 
            centrosymmetric groups, while origin choice 2 in BCS.
        search_cell : bool, default=False
            Whether the transformation to the conventional cell should be computed.
            It is `True` if kpnames was specified in CLI.
        trans_thresh : float, default=1e-5
            Threshold to compare translational parts of symmetries.
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing

        Returns
        -------
        array 
            Transformation of vectors defining the unit cell to the 
            standard setting.
        array
            Shift taking the origin of the unit cell used in the DFT 
            calculation to that of the convenctional setting used in 
            BCS.

        Raises
        ------
        RuntimeError
            Could not find a pait (refUC,shiftUC) matching symmetries 
            obtained from spglib to those in tables (mod. lattice 
            translations of the primitive cell).

        """
        # Give preference to CLI input
        refUC_cli_bool = refUC_cli is not None
        shiftUC_cli_bool = shiftUC_cli is not None
        if refUC_cli_bool and shiftUC_cli_bool:  # Both specified in CLI.
            refUC = refUC_cli.T  # User sets refUC as if it was acting on column
            shiftUC = shiftUC_cli
            log_message('refUC and shiftUC read from CLI', verbosity, 1)
            return refUC, shiftUC
        elif refUC_cli_bool and not shiftUC_cli_bool:  # shiftUC not given in CLI.
            refUC = refUC_cli.T  # User sets refUC as if it was acting on column
            shiftUC = np.zeros(3, dtype=float)
            log_message('refUC was specified in CLI, but shiftUC was not.'
                        ' Taking shiftUC=(0,0,0)', verbosity, 1)
            return refUC, shiftUC
        elif not refUC_cli_bool and shiftUC_cli_bool:  # refUC not given in CLI.
            refUC = np.eye(3, dtype=float)
            shiftUC = shiftUC_cli
            log_message('shitfUC was specified in CLI, but refUC was not. Taking '
                        '3x3 identity matrix as refUC.', verbosity, 1)
            return refUC, shiftUC
        elif not search_cell:
            refUC = np.eye(3, dtype=float)
            shiftUC = np.zeros(3, dtype=float)
            log_message('Taking 3x3 identity matrix as refUC and shiftUC=(0,0,0). '
                        'If you want to calculate the transformation to '
                        'conventional cell, run IrRep with -searchcell', verbosity, 1)
            return refUC, shiftUC
        else:  # Neither specifiend in CLI.
            log_message('Determining transformation to conventional setting '
                        '(refUC and shiftUC)', verbosity, 1)
            refUC = np.linalg.inv(refUC_lib)  # from DFT to convenctional cell

            # Check if the shift given by spglib works
            shiftUC = -refUC.dot(shiftUC_lib)
            try:
                ind, dt, signs, U = self.match_symmetries(
                    refUC,
                    shiftUC,
                    trans_thresh=trans_thresh
                )
                return refUC, shiftUC
            except RuntimeError:
                pass

            # Check if the group is centrosymmetric
            inv = None
            for sym in self.u_symmetries:
                if np.allclose(sym.rotation, -np.eye(3)):
                    inv = sym

            if inv is None:  # Not centrosymmetric
                for r_center in self.vecs_centering():
                    shiftUC = shiftUC_lib + refUC.dot(r_center)
                    try:
                        ind, dt, signs, _ = self.match_symmetries(
                            refUC,
                            shiftUC,
                            trans_thresh=trans_thresh,
                            only_u_symmetries=True
                        )
                        log_message(f'ShiftUC achieved with the centering: {r_center}',
                                    verbosity, 1)
                        return refUC, shiftUC
                    except RuntimeError:
                        pass
                raise RuntimeError(("Could not find any shift that leads to "
                                    "the expressions for the symmetries found "
                                    "in the tables."))

            else:  # Centrosymmetric. Origin must sit in an inv. center
                for r_center in self.vecs_inv_centers():
                    shiftUC = 0.5 * inv.translation + refUC.dot(0.5 * r_center)
                    try:
                        ind, dt, signs, _ = self.match_symmetries(refUC,
                                                                  shiftUC,
                                                                  trans_thresh=trans_thresh,
                                                                  only_u_symmetries=True
                                                                  )
                        log_message(f"ShiftUC achieved in 2 steps:\n"
                                    f"  (1) Place origin of primitive cell on inversion center: {0.5 * inv.translation}\n"
                                    f"  (2) Move origin of convenctional cell to the inversion-center: {r_center}",
                                    verbosity, 1)
                        return refUC, shiftUC
                    except RuntimeError:
                        pass
                raise RuntimeError(("Could not find any shift that places the "
                                    "origin on an inversion center which leads "
                                    "to the expressions for the symmetries "
                                    "found in the tables. Enter refUC and "
                                    "shiftUC in command line"))



    def match_symmetries(
            self,
            refUC=None,
            shiftUC=None,
            signs=False,
            trans_thresh=1e-5,
            only_u_symmetries=False
    ):
        """
        Matches symmetry operations of two lists. Translational parts 
        are matched mod. lattice translations (important for centered 
        structures).

        Parameters
        ----------
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=None
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        signs : bool, default=False
            If `True`, match also rotations of spinors corresponding to 
            each symmetry.
        trans_thresh : float, default=1e-5
            Threshold used to compare translational parts of symmetries.
        only_y_symmetries: boolean, default=False
            Only match unitary symmetries. Useful when determining the centering

        Returns
        -------
        list
            The :math:`i^{th}` element corresponds to the :math:`i^{th}` 
            symmetry found by `spglib` and it is the position that the 
            same symmetry has in the tables.
        list
            The :math:`i^{th}` element corresponds to the :math:`i^{th}` 
            symmetry found by `spglib` and it is the phase difference 
            w.r.t. the same symmetry in the tables, which may arise if 
            their translational parts differ by a lattice vector.
        array
            The :math:`i^{th}` element corresponds to the :math:`i^{th}` 
            symmetry found by `spglib` and it is the sign needed to make
            the matrix for spinor rotation identical to that in tables.
        """

        if refUC is None:
            refUC = self.refUC
        if shiftUC is None:
            shiftUC = self.shiftUC

        ind = []
        dt = []
        if only_u_symmetries:
            symmetries = self.u_symmetries
            symmetries_tables = self.u_symmetries_tables
        else:
            symmetries = self.symmetries
            symmetries_tables = self.u_symmetries_tables + self.au_symmetries_tables


        for j, sym in enumerate(symmetries):
            R = sym.rotation_refUC(refUC)
            t = sym.translation_refUC(refUC, shiftUC)
            found = False
            for i, sym2 in enumerate(symmetries_tables):
                t1 = refUC.dot(sym2.t - t) % 1
                # t1 = np.dot(sym2.t - t, refUC) % 1
                t1[1 - t1 < trans_thresh] = 0
                if np.allclose(R, sym2.R) and sym.time_reversal == sym2.time_reversal:
                    if np.allclose(t1, [0, 0, 0], atol=trans_thresh):
                        ind.append(i)  # au symmetries labeled from 0
                        dt.append(sym2.t - t)
                        found = True
                        break
                        # Tolerance for rotational part comparison
                        # is much more restrictive than for transl.
                        # Make them consistent?
                    else:
                        raise RuntimeError(
                            f"Error matching translational part for symmetry {j + 1}. "
                            f"A symmetry with identical rotational part has been found in tables, "
                            f"but their translational parts do not match:\n"
                            f"R (found, in conv. cell)= \n{R}\n"
                            f"t(found) = {sym.translation}\n"
                            f"t(table) = {sym2.t}\n"
                            f"t(found, in conv. cell) = {t}\n"
                            f"t(table)-t(found) (in conv. cell, mod. lattice translation)= {t1}"
                        )
            if not found:
                raise RuntimeError(
                    f"Error matching rotational part for symmetry {j + 1}. "
                    f"In the tables there is not any symmetry with identical rotational part.\n"
                    f"R(found) = \n{R}\nt(found) = {t}"
                )

        order = len(symmetries)
        if len(set(ind)) != order:
            raise RuntimeError(
                "Error in matching symmetries detected by spglib with the \
                 symmetries in the tables. Try to modify the refUC and shiftUC \
                 parameters")

        if signs:
            S1 = [sym.spinor_rotation for sym in symmetries]
            S2 = [symmetries_tables[i].S for i in ind]
            signs_array, U = match_spinor_rotations(S1, S2)
        else:
            signs_array = np.ones(len(ind), dtype=int)
            U = None
        return ind, dt, signs_array, U

    def vecs_centering(self):
        """ 
        Check the space group and generate vectors of centering.

        Returns
        -------
        array
            Each row is a lattice vector describing the centering.
        """
        cent = np.array([[0, 0, 0]])
        if self.name[0] == 'P':
            pass  # Just to make it explicit
        elif self.name[0] == 'C':
            cent = np.vstack((cent, cent + [1 / 2, 1 / 2, 0]))
        elif self.name[0] == 'I':
            cent = np.vstack((cent, cent + [1 / 2, 1 / 2, 1 / 2]))
        elif self.name[0] == 'F':
            cent = np.vstack((cent,
                              cent + [0, 1 / 2, 1 / 2],
                              cent + [1 / 2, 0, 1 / 2],
                              cent + [1 / 2, 1 / 2, 0],
                              )
                             )
        elif self.name[0] == 'A':  # test this
            cent = np.vstack((cent, cent + [0, 1 / 2, 1 / 2]))
        else:  # R-centered
            cent = np.vstack((cent,
                              cent + [2 / 3, 1 / 3, 1 / 3],
                              cent + [1 / 3, 2 / 3, 2 / 3],
                              )
                             )
        return cent

    def vecs_inv_centers(self):
        """
        Get the positions of all inversion centers in the unit cell.

        Returns
        -------
        array
            Each element is a vector pointing to a position center in 
            the convenctional unit cell.

        Notes
        -----
        If the space group is primitive, there are 8 inversion centers, 
        but there are more if it is a group with a centering.

        """
        vecs = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]
            ]
        )
        vecs = np.vstack([vecs + r for r in self.vecs_centering()])
        return vecs

    def print_hs_kpoints(self):
        """
        Give the kpoint coordinates of the symmetry tables transformed to 
        the DFT calculation cell.

        """

        table = IrrepTable(self.number_str, self.spinor, magnetic=self.magnetic)
        refUC_kspace = np.linalg.inv(self.refUC.T)

        matrix_format = ("\t\t| {: .2f} {: .2f} {: .2f} |\n"
                         "\t\t| {: .2f} {: .2f} {: .2f} |\n"
                         "\t\t| {: .2f} {: .2f} {: .2f} |\n\n")

        print("\n---------- HS-KPOINTS FOR IRREP IDENTIFICATION ----------\n")

        print("\nChange of coordinates from conventional to DFT cell:\n")
        print(matrix_format.format(*refUC_kspace.ravel()))

        print("\nChange of coordinates from DFT to conventional cell:\n")
        print(matrix_format.format(*np.linalg.inv(refUC_kspace).ravel()))

        _, kp_index = np.unique([irr.kpname for irr in table.irreps], return_index=True)
        print("Coordinates in symmetry tables:\n")
        for i in kp_index:
            name = table.irreps[i].kpname
            coords = table.irreps[i].k
            print("\t {:<2} : {: .6f} {: .6f} {: .6f}".format(name, *coords))
        print("\nCoordinates for DFT calculation:\n")
        for i in kp_index:
            name = table.irreps[i].kpname
            coords = table.irreps[i].k
            k_dft = np.round(refUC_kspace.dot(coords), 6) % 1
            print("\t {:<2} : {: .6f} {: .6f} {: .6f}".format(name, *k_dft))

    def kpoints_to_calculation_cell(self, kpoints):
        """Transforms kpoints form standard cell to calculation cell

        Parameters
        ----------
        kpoints : np.NDArray
            kpoints in standard cell
        """
        refUC_kspace = np.linalg.inv(self.refUC.T)

        kpoints = np.array([refUC_kspace.dot(k) for k in kpoints]) % 1

        return kpoints

    def kpoints_to_standard_cell(self, kpoints):
        """Transforms kpoints form standard cell to calculation cell

        Parameters
        ----------
        kpoints : np.NDArray
            kpoints in standard cell
        """
        refUC_kspace = self.refUC.T

        kpoints = np.array([refUC_kspace.dot(k) for k in kpoints]) % 1

        return kpoints



def match_spinor_rotations(S1, S2):
    """
    Determine the sign difference between matrices describing the 
    transformation of spinors found by `spglib` and those read from tables.

    Parameters
    ----------
    S1 : list
        Contains the matrices for the transformation of spinors 
        corresponding to symmetry operations found by `spglib`.
    S2 : list
        Contains the matrices for the transformation of spinors 
        corresponding to symmetry operations read from tables.

    Returns
    -------
    array
        The `j`-th element is the matrix to match the `j`-th matrices of 
        `S1` and `S2`.
    """
    n = 2

    def RR(x):
        """
        Constructs a 2x2 complex matrix out of a list containing real and 
        imaginary parts.

        Parameters
        ----------
        x : list, length=8
            Length is 8. `x[:4]` contains the real parts, `x[4:]` the 
            imaginary parts.

        Returns
        -------
        array, shape=(2,2)
            Matrix of complex elements. 
        """
        return np.array([[x1 + 1j * x2 for x1, x2 in zip(l1, l2)] for l1, l2 in zip(x[:n * n].reshape((n, n)), x[n * n:].reshape((n, n)))])

    def residue_matrix(r):
        """
        Calculate the residue of a matrix.

        Parameters
        ----------
        r : array
            Matrix used as ansatz for the minimization.

        Returns
        -------
        float            
        """

        return sum([min(abs(r.dot(b).dot(r.T.conj()) - s * a).sum() for s in (1, -1)) for a, b in zip(S1, S2)])

    def residue(x):
        """
        Calculate the normalized residue.

        Parameters
        ----------
        x : list, length=8
            Length is 8. `x[:4]` contains the real parts, `x[4:]` the 
            imaginary parts.

        Returns
        -------
        float
        """
        return residue_matrix(RR(x)) / len(S1)

    for i in range(11):
        x0 = np.random.random(2 * n * n)
        res = minimize(residue, x0)
        r = res.fun
        if r < 1e-4:
            break
    if r > 1e-3:
        raise RuntimeError(
            "the accurcy is only {0}. Is this good?".format(r))

    R1 = RR(res.x)
    signs = np.array([R1.dot(b).dot(R1.T.conj()).dot(np.linalg.inv(a)).diagonal().mean().real.round() for a, b in zip(S1, S2)], dtype=int)

    return signs, R1
