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


from functools import cached_property
import warnings
import numpy as np
from scipy.linalg import expm
from .utility import UniqueListMod1, str_, BOHR, cached_einsum
from .gvectors import transform_gk

pauli_sigma = np.array(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


class SymmetryOperation():
    """
    Contains information to describe a symmetry operation and methods to get 
    info about the symmetry, transform it to a reference choice of unit cell 
    and print a description of it.

    Parameters
    ----------
    rotation : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    translation : array, shape=(3,)
        Translational part of the symmetry operation, in terms of the basis 
        vectors of the unit cell.
    time_reversal : bool, default=False
        `True` if the symmetry operation includes time-reversal.
    Lattice : array, shape=(3,3) 
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    ind : int, default=-1
        Index of the symmetry operation.
    spinor : bool, default=true
        `True` if wave-functions are spinors, `False` if they are scalars.
    translation_mod1 : bool, default=True
        If `True`, the translation part of the symmetry operation is taken
        modulo 1. Otherwise, it is taken as it is

    Attributes
    ---------
    ind : int
        Index of the symmetry operation.
    rotation : array, shape=(3,3)
        Matrix describing the tranformation of basis vectors of the unit cell 
        under the symmetry operation.
    translation : array, shape=(3,)
        Translational part of the symmetry operation, in terms of the basis 
        vectors of the unit cell.
    time_reversal : bool, default=False
        `True` if the symmetry operation includes time-reversal.
    Lattice : array, shape=(3,3) 
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    axis : array, shape=(3,)
        Rotation axis of the symmetry.
    angle : float
        Rotation angle of the symmmetry, in radians.
    inversion : bool
        `False` if the symmetry preserves handedness (identity, rotation, 
        translation or screw rotation), `True` otherwise (inversion, reflection 
        roto-inversion or glide reflection).
    angle_str : str
        String describing the rotation angle in radians.
    spinor : bool
        `True` if wave-functions are spinors, `False` if they are scalars.
    spinor_rotation : array, shape=(2,2)
        Matrix describing how spinors transform under the symmetry.
    sign : float
        Factor needed to match the matrix for the rotation of spinors 
        to that in tables.
    """

    def __init__(self, rot, trans, Lattice, time_reversal=False, ind=-1, spinor=True,
                 positions=None,
                 translation_mod1=True, spinor_rotation=None):
        self.ind = ind
        self.rotation = rot
        self.positions = positions
        self.time_reversal = bool(time_reversal)
        self.real_lattice = Lattice
        self.translation_mod1 = translation_mod1
        self.translation = self.get_transl_mod1(trans)
        self.axis, self.angle, self.inversion = self._get_operation_type()
        iangle = (round(self.angle / np.pi * 6) + 6) % 12 - 6
        if iangle == -6:
            iangle = 6
        self.angle = iangle * np.pi / 6
        self.angle_str = self.get_angle_str()
        self.spinor = spinor
        if spinor_rotation is None:
            self.spinor_rotation = expm(-0.5j * self.angle *
                                    cached_einsum('i,ijk->jk', self.axis, pauli_sigma))
        else:
            self.spinor_rotation = spinor_rotation
        self.sign = 1  # May be changed later externally

    def inverse(self):
        """
        Create the inverse symmetry operation.

        Returns
        -------
        SymmetryOperation
            A new instance of the symmetry operation representing the inverse.
        """
        rot_inv = np.linalg.inv(self.rotation)
        trans_inv = -rot_inv @ self.translation
        spinor_rot_inv = self.spinor_rotation.conj().T
        if self.time_reversal:
            warnings.warn("The inverse of a symmetry with time-reversal is not well defined.")
            spinor_rot_inv = np.array([[0, -1], [1, 0]]) @ spinor_rot_inv
        return SymmetryOperation(rot=rot_inv,
                                 trans=trans_inv,
                                 Lattice=self.real_lattice,
                                 time_reversal=self.time_reversal,
                                 ind=self.ind,
                                 spinor=self.spinor,
                                 translation_mod1=self.translation_mod1,
                                 spinor_rotation=spinor_rot_inv)

    def __mul__(self, other):
        """
        Create the product of two symmetry operations.

        Parameters
        ----------
        other : SymmetryOperation
            The symmetry operation to multiply with.

        Returns
        -------
        SymmetryOperation
            A new instance of the symmetry operation representing the product.
        """
        return self.multiply_keeptransl(other, mod1=self.translation_mod1)

    def multiply_keeptransl(self, other, mod1=False):
        """
        Create the product of two symmetry operations, possibly without taking the translational part modulo 1.

        Parameters
        ----------
        other : SymmetryOperation
            The symmetry operation to multiply with.
        mod1 : bool, default=False
            If `True`, the translational part is taken modulo 1.

        Returns
        -------
        SymmetryOperation
            A new instance of the symmetry operation representing the product.
        """
        if not isinstance(other, SymmetryOperation):
            raise RuntimeError("Can only multiply two SymmetryOperation objects.")
        rot_new = self.rotation @ other.rotation
        trans_new = self.translation + self.rotation @ other.translation
        if self.spinor and other.spinor:
            spinor_rot_new = self.spinor_rotation @ other.spinor_rotation
            # Time reversal squares to -1 for spinors, and commutes with rotations
            if self.time_reversal and other.time_reversal:
                spinor_rot_new = -spinor_rot_new
        else:
            spinor_rot_new = None
        return SymmetryOperation(rot=rot_new,
                                 trans=trans_new,
                                 Lattice=self.real_lattice,
                                 time_reversal=self.time_reversal != other.time_reversal,
                                 ind=self.ind,
                                 spinor=self.spinor,
                                 translation_mod1=mod1,
                                 spinor_rotation=spinor_rot_new)

    def equals(self, other, mod1=True, tol=1e-6):
        """
        Check if two symmetry operations are equal.

        Parameters
        ----------
        other : SymmetryOperation
            The symmetry operation to compare with.
        mod1 : bool, default=True
            If `True`, the translation part is compared modulo 1.

        Returns
        -------
        bool
            `True` if the symmetry operations are equal, `False` otherwise.
        """
        if not isinstance(other, SymmetryOperation):
            raise ValueError("Can only compare two SymmetryOperation objects.")
        if not np.all(self.rotation == other.rotation):
            return False
        if self.time_reversal != other.time_reversal:
            return False
        dt = self.translation - other.translation
        if mod1:
            dt = dt - np.round(dt)
        if not np.all(np.abs(dt) < tol):
            return False
        return True


    @property
    def lattice(self):
        return self.real_lattice

    @property
    def Lattice(self):
        """
        Lattice vectors in the DFT cell setting. For backward compatibility.
        """
        return self.real_lattice


    def get_transl_mod1(self, t):
        """
        Take translation modulo 1 if needed.
        governed by the translation_mod1 attribute.
        """
        if self.translation_mod1:
            t = t % 1
            t[1 - t < 1e-5] = 0
            return t
        else:
            return t

    def get_angle_str(self):
        """
        Give str of rotation angle.

        Returns
        -------
        str
            Rotation angle in radians.

        Raises
        ------
        RuntimeError
            Angle does not belong to 1, 2, 3, 4 or 6-fold rotation.
        """
        accur = 1e-4

        def is_close_int(x):
            return abs((x + 0.5) % 1 - 0.5) < accur

        api = self.angle / np.pi
        if abs(api) < 0.01:
            return " 0 "
        for n in 1, 2, 3, 4, 6:
            if is_close_int(api * n):
                return "{num:.0f}{denom} pi".format(
                    num=round(api * n), denom="" if n == 1 else "/" + str(n))
        raise RuntimeError(f"{api} pi rotation cannot be in the space group")

    @cached_property
    def is_identity(self):
        return np.allclose(self.rotation, np.eye(3)) and np.allclose(self.translation, 0) and not self.time_reversal and not self.inversion


    def _get_operation_type(self):
        """
        Calculates the rotation axis and angle of the symmetry and if it 
        preserves handedness or not.

        Returns
        -------
        tuple
            The first element is an array describing the rotation axis. The 
            second element describes the rotation angle. The third element is a 
            boolean, `True` if the symmetry preserves handedness 
            (determinant -1).
        """
        rotxyz = self.rotation_cart
#        print ("rotation in real space:\n",rotxyz)
        E, V = np.linalg.eig(rotxyz)
        if not np.isclose(abs(E), 1).all():
            raise RuntimeError(
                "some eigenvalues of the rotation are not unitary")
        if E.prod() < 0:
            inversion = True
            E *= -1
        else:
            inversion = False
        idx = np.argsort(E.real)
        E = E[idx]
        V = V[:, idx]
        axis = V[:, 2].real
        if np.isclose(E[:2], 1).all():
            angle = 0
        elif np.isclose(E[:2], -1).all():
            angle = np.pi
        else:
            angle = np.angle(E[0])
            v = V[:, 0]
            s = np.real(np.linalg.det([v, v.conj(), axis]) / 1.j)
            if np.isclose(s, -1):
                angle = 2 * np.pi - angle
            elif not np.isclose(s, 1):
                raise RuntimeError(f"the sign of rotation should be +-1, cannot determine for {rotxyz}")
        return (axis, angle, inversion)

    def rotation_refUC(self, refUC):
        """
        Calculate the matrix of the symmetry in the reference cell choice.

        Parameters
        ----------
        refUC : array
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.

        Returns
        -------
        R1 : array, shape=(3,3)
            Matrix for the transformation of basis vectors forming the 
            reference unit cell.

        Raises
        ------
        RuntimeError
            If the matrix contains non-integer elements after the transformation.
        """
        R = np.linalg.inv(refUC).dot(self.rotation).dot(refUC)
        R1 = np.array(R.round(), dtype=int)
        if (abs(R - R1).max() > 1e-6):
            raise RuntimeError(f"the rotation in the reference UC is not integer. Is that OK? \n{R}")
        return R1

    def translation_refUC(self, refUC, shiftUC):
        """
        Calculate translation in reference choice of unit cell.

        Parameters
        ----------
        refUC : array
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.

        Returns
        -------
        array
            Translation in reference choice of unit cell.
        """
        t_ref = - shiftUC + self.translation + self.rotation.dot(shiftUC)
        t_ref = np.linalg.inv(refUC).dot(t_ref)
        return t_ref

    def spinrotation_refUC(self, U):
        """
        Calculate spin representation matrix in reference cell

        Parameters
        ----------
        U : array
            Unitary transformation of spin quantization axis from spglib's 
            choice to reference cell's choice

        Returns
        -------
        array
            Spin representation matrix in reference cell
        """

        S = U.conj().transpose() @ self.spinor_rotation @ U
        S *= self.sign
        return S


    def show(self, refUC=np.eye(3), shiftUC=np.zeros(3), U=np.eye(2)):
        """
        Print description of symmetry operation.

        Parameters
        ----------
        refUC : array, default=np.eye(3)
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=np.zeros(3)
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        U : array, default=np.zeros(2)
            Unitary transformation of spin quantization axis from spglib's 
            choice to reference cell's choice
        """

        def parse_row_transform(mrow):
            s = ""
            coord = ["kx", "ky", "kz"]
            is_first = True
            for i in range(len(mrow)):
                b = int(mrow[i]) if np.isclose(mrow[i], int(mrow[i])) else mrow[i]
                if b == 0:
                    continue
                if b == 1:
                    if is_first:
                        s += coord[i]
                    else:
                        s += "+" + coord[i]
                elif b == -1:
                    s += "-" + coord[i]
                else:
                    if b > 0:
                        s += "+" + str(b) + coord[i]
                    else:
                        s += str(b) + coord[i]
                is_first = False
            return s

        if refUC is None or shiftUC is None:
            write_ref = False
        elif (not np.allclose(refUC, np.eye(3)) or
                not np.allclose(shiftUC, np.zeros(3))):
            write_ref = True  # To avoid writing this huge block again
        else:
            write_ref = False

        # Print header
        print(f"\n ### {self.ind} \n")

        # Print rotation part
        rotstr = [
            f"{s}{' '.join(f'{x:3d}' for x in row)}{t}"
            for s, row, t in zip(
                ["rotation : |", " " * 11 + "|", " " * 11 + "|"],
                self.rotation,
                [" |", " |", " |"]
            )
        ]
        if write_ref:
            R = self.rotation_refUC(refUC)
            rotstr1 = [
                f"{' ' * 5}{s}{' '.join(f'{x:3d}' for x in row)}{t}"
                for s, row, t in zip(
                    ["rotation : |", " (refUC)   |", " " * 11 + "|"],
                    R,
                    [" |", " |", " |"]
                )
            ]
            rotstr = [r + r1 for r, r1 in zip(rotstr, rotstr1)]

        matrix = np.transpose(np.linalg.inv(self.rotation))
        if self.time_reversal:
            matrix *= -1
        kstring = "gk = [" + ", ".join(
            [parse_row_transform(r) for r in matrix]
        ) + "]"


        if write_ref:
            matrix = np.transpose(np.linalg.inv(R))
            if self.time_reversal:
                matrix *= -1
            kstring += "  |   refUC:  gk = [" + ", ".join(
                [parse_row_transform(r) for r in matrix]
            ) + "]"

        print("\n".join(rotstr))
        print("\n\n", kstring)

        # Print spinor transformation matrix
        if self.spinor:
            spinstr = [f"{s}{' '.join(f'{x.real:6.3f}{x.imag:+6.3f}j' for x in row)}{t}"
                       for s, row, t in zip(
                           ["\nspinor rot.         : |", " " * 22 + "|"],
                           self.spinor_rotation,
                           [" |", " |"])
            ]
            print("\n".join(spinstr))
            if write_ref:
                spinstr = [s + " ".join(f"{x.real:6.3f}{x.imag:+6.3f}j" for x in row) + t
                           for s, row, t in zip(["spinor rot. (refUC) : |", " " * 22 + "|",],
                                                self.spinrotation_refUC(U),
                                                [" |", " |"])
                            ]
                print("\n".join(spinstr))

        # Print translation part
        trastr = ("\ntranslation         :  [ " +
                  " ".join(f"{x:8.4f}" for x in self.get_transl_mod1(self.translation.round(6))) +
                  " ] ")
        print(trastr)

        if write_ref:
            _t = self.translation_refUC(refUC, shiftUC)
            trastr = f"translation (refUC) :  [ {' '.join(f'{x:8.4f}' for x in self.get_transl_mod1(_t.round(6)))} ] "
            print(trastr)

        print(f"\naxis: {self.axis.round(6)} ; angle = {self.angle_str}, "
              f"inversion: {self.inversion}, time reversal: {self.time_reversal}")

    def copy(self):
        """
        Create a copy of the symmetry operation.

        Returns
        -------
        SymmetryOperation
            A new instance of the symmetry operation with the same attributes.
        """
        return SymmetryOperation(
            self.rotation.copy(),
            self.translation.copy(),
            self.real_lattice.copy(),
            self.time_reversal,
            self.ind,
            self.spinor,
            self.translation_mod1,
            self.spinor_rotation.copy()
        )

    def str(self, refUC=np.eye(3), shiftUC=np.zeros(3)):
        """
        Construct description of symmetry operation.

        Parameters
        ----------
        refUC : array, default=np.eye(3)
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=np.zeros(3)
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.

        Returns
        -------
        str
            Description to print.
        """
#        print ( "symmetry # ",self.ind )
        R = self.rotation_refUC(refUC)
        t = self.translation_refUC(refUC, shiftUC)
#        np.savetxt(stdout,np.hstack( (R,t[:,None])),fmt="%8.5f" )
        S = self.spinor_rotation
        return ("   ".join(" ".join(str(x) for x in r) for r in R) + "     " + " ".join(str_(x) for x in t) + ("      " +
                "    ".join("  ".join(str_(x) for x in X) for X in (np.abs(S.reshape(-1)), np.angle(S.reshape(-1)) / np.pi))) +
                f"\n time-reversal : {self.time_reversal} \n")

    def str2(self, refUC=np.eye(3), shiftUC=np.zeros(3), write_tr=False):
        """
        Print matrix of a symmetry operation in the format: 
        {{R|t}}-> R11,R12,...,R23,R33,t1,t2,t3 and, when SOC was included, the 
        elements of the matrix describing the transformation of the spinor in 
        the format:
        Re(S11),Im(S11),Re(S12),...,Re(S22),Im(S22).

        Parameters
        ----------
        refUC : array, default=np.eye(3)
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=np.zeros(3)
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.

        Returns
        -------
        str
            Description to print.
        """

        if refUC is None:
            refUC = np.eye(3, dtype=int)
        if shiftUC is None:
            shiftUC = np.zeros(3, dtype=float)
        # this method for Bilbao server
        #       refUC - row-vectors, expressing the reference unit cell vectors in terms of the lattice used in calculation
        #        print ( "symmetry # ",self.ind )
        R = self.rotation
        t = self.translation
        S = self.spinor_rotation
        tr = -1 if self.time_reversal else 1
        sR = " ".join(f"{x:>2d}" for row in R for x in row)
        st = " ".join(f"{x: >10.6f}" for x in t)

        if S is not None:
            sS = " ".join(f"{x.real: >10.6f} {x.imag: >10.6f}" for x in S.reshape(-1))
        else:
            sS = ""

        if write_tr:
            return f"{sR}     {st}{sS} {tr}\n"
        else:
            return f"{sR}     {st}{sS}\n"

    def str_sym(self, alat):
        """
        Write 4 strings (+1 empty) for the prefix.sym file 
        for sitesym in wannier90: 
        The symmetry operations act on a point r as rR − t.

        Parameters
        ----------
        alat : float
            Lattice parameter in angstroms.

        Returns
        -------
        str
            Description to print.
            1 blank line
            3 lines: cartesian rotation matrix
            1 line : cartesian translation in units of alat
        """

        Rcart = self.real_lattice.T.dot(self.rotation).dot(np.linalg.inv(self.real_lattice).T)
        t = - self.translation @ self.real_lattice / alat / BOHR

        arr = np.vstack((Rcart, [t]))
        return "\n" + "".join("   ".join(f"{x:20.15f}" for x in r) + "\n" for r in arr)

    def json_dict(self, refUC=np.eye(3), shiftUC=np.zeros(3)):
        '''
        Prepare dictionary with info of symmetry to save in JSON

        Returns
        -------
        d : dict
            Dictionary with info about symmetry
        '''

        d = {}
        d["axis"] = self.axis
        d["angle str"] = self.angle_str
        d["angle pi"] = self.angle / np.pi
        d["inversion"] = self.inversion
        d["sign"] = self.sign

        d["rotation matrix"] = self.rotation
        d["translation"] = self.translation

        R = self.rotation_refUC(refUC)
        t = self.translation_refUC(refUC, shiftUC)
        d["rotation matrix refUC"] = R
        d["translation refUC"] = t

        return d

    def transform_r(self, vector, inverse=False):
        """
        Transform a real-space vector (in lattice coordinates) under the symmetry operation.

        Parameters
        ----------
        vector : array((...,3), dtype=float) 
            Vector to transform. (or array of vectors)

        Returns
        -------
        array
            Transformed vector.
        """
        if inverse:
            return (np.array(vector) - self.translation[..., :]).dot(self.rotation_inv.T)
        else:
            return np.array(vector).dot(self.rotation.T) + self.translation[..., :]


    @cached_property
    def rotation_cart(self):
        """
        Calculate the rotation matrix in cartesian coordinates.
        """
        return self.real_lattice.T @ self.rotation @ self.lattice_inv.T

    @cached_property
    def translation_cart(self):
        return self.real_lattice.T @ self.translation @ self.lattice_inv.T

    @cached_property
    def lattice_inv(self):
        return np.linalg.inv(self.real_lattice)

    @cached_property
    def reciprocal_lattice(self):
        return self.lattice_inv.T

    @cached_property
    def det_cart(self):
        return np.linalg.det(self.rotation_cart)

    @cached_property
    def det(self):
        return np.linalg.det(self.rotation)


    @cached_property
    def rotation_inv(self):
        rotinv = np.linalg.inv(self.rotation)
        rotinv_int = np.array(rotinv.round(), dtype=int)
        assert np.allclose(rotinv, rotinv_int, atol=1e-6), f"Inverse Rotation matrix is not integer: {rotinv}"
        return rotinv_int

    @cached_property
    def spinor_rotation_TR(self):
        """
        Calculate the spinor rotation matrix under time-reversal.

        Returns
        -------
        array
            Spinor rotation matrix under time-reversal.
        """
        if self.time_reversal:
            return np.array([[0, 1], [-1, 0]]) @ self.spinor_rotation.conj()
        else:
            return self.spinor_rotation

    def transform_WF(self, k, WF, igall, k_new=None):
        """
        Transform wavefunction under the symmetry operation.

        Parameters
        ----------
        k: array((3,), dtype=float)
            k-point to transform. (in reduced coordinates)
        WF : array(nb,ng*nspinor, dtype=complex)
            Wavefunction to transform. (or array of wavefunctions)
        igall : np.ndarray((ng, 6), dtype=int)
            the array 
        k_new : array((3,), dtype=float), optional
            the new k-point to transform to. if provided, it will be checked that the
            transformation is consistent with the new k-point (modulo reciprocal lattice vectors).
            If not provided, the new k-point will be the transformed k-point.

        Returns
        -------
        k_new : array((3,), dtype=float)
            the transformed k-point.
        """
        # TODO : check all signs and other details
        k_new, igTr = self.transform_gk(k, igall, k_other=k_new)

        igall_new = np.copy(igall)
        igall_new[:, :3] = igTr

        multZ = np.exp(-2j * np.pi * (igall_new[:, :3] + k_new[None, :]) @ self.translation)
        if self.time_reversal:
            WF = WF.conj()
        WF = WF[:, :, :] * multZ[None, :, None]
        if self.spinor:
            WF = cached_einsum("ts,mgs->mgt", self.spinor_rotation_TR, WF)
        return k_new, WF, igall_new

    def transform_gk(self, k, ig, k_other=None):
        A = self.rotation
        if self.time_reversal:
            A = -A
        return transform_gk(k, ig, A, kpt_other=k_other)

    def transform_k(self, vector, inverse=False):
        """
        Transform a k-space vector under the symmetry operation.

        Parameters
        ----------
        vector : array((...,3), dtype=float) 
            Vector to transform. (or array of vectors)

        Returns
        -------
        array
            Transformed vector.
        """
        # print (f"rotation = {self.rotation}")
        # print (f"transforming {vector} ({vector.dot(self.reciprocal_lattice)})")
        # print (f"by {self.rotation_inv}")
        if inverse:
            res = vector.dot(self.rotation)
        else:
            res = vector.dot(self.rotation_inv)
        # print (f"got {res} ({res.dot(self.reciprocal_lattice)})")
        if self.time_reversal:
            res = -res
        return res

    def transform_grid_indices(self, size, inverse=False):
        """
        Transform a grid of k-points under the symmetry operation.

        Parameters
        ----------
        size : array((3,), dtype=int)
            Size of the grid in each direction.

        Returns
        -------
        array
            Transformed grid.
        """
        size = tuple(size)
        assert len(size) == 3
        if not hasattr(self, 'rotate_grid_cache'):
            self.rotate_grid_cache = {}
        if size not in self.rotate_grid_cache:
            # First compute direct transformation
            i_rot = self.rotation
            i_trans = self.translation * np.array(size)
            i_trans_int = np.round(i_trans).astype(int)
            ind_original = np.indices(size).reshape((3, -1))
            assert np.allclose(i_trans, i_trans_int), f"Translation {self.translation} not compatible with grid {size}"
            indx = np.dot(i_rot, ind_original) + i_trans_int[:, None]
            indx = np.ravel_multi_index(indx, size, 'wrap')
            # now compute inverse transformation
            i_rot_inv = self.rotation_inv
            indx_inv = np.dot(i_rot_inv, ind_original - i_trans_int[:, None])
            indx_inv = np.ravel_multi_index(indx_inv, size, 'wrap')
            size_tot = np.prod(size)
            assert np.all(np.sort(indx) == np.arange(size_tot)), f"Rotation does not permute the grid correctly: {indx}"
            assert np.all(np.sort(indx_inv) == np.arange(size_tot)), f"Inverse Rotation does not permute the grid correctly: {indx_inv}"
            assert np.all(indx[indx_inv] == np.arange(size_tot)), f"Inverse rotation is not consistent with direct rotation: {indx} and {indx_inv}, {indx[indx_inv]}"
            self.rotate_grid_cache[size] = {True: indx_inv, False: indx}
        return self.rotate_grid_cache[size][inverse]

    def transform_grid_data(self, grid_data, inverse=False):
        if grid_data.ndim < 3:
            raise ValueError("data should have shape (...,Nx,Ny,Nz)")
        Nc = grid_data.shape[-3:]
        shape_front = grid_data.shape[:-3]
        Nc_tot = np.prod(Nc)
        NB = int(grid_data.size / Nc_tot)
        assert NB * Nc_tot == grid_data.size, f"the data should have shape (...,Nx,Ny,Nz), but got {grid_data.shape}"
        indx = self.transform_grid_indices(Nc, inverse=False)
        grid_data = grid_data.reshape(NB, Nc_tot)
        grid_data_new = np.zeros(grid_data.shape, dtype=grid_data.dtype)
        grid_data_new[:, indx] = grid_data
        return grid_data_new.reshape(shape_front + Nc)



    def set_R_aii_gpaw(self, calc, reset=False):
        """rotation matrices for the PAW projections."""
        if not hasattr(self, 'R_aii') or reset:
            from gpaw.atomrotations import AtomRotations
            setups = calc.setups

            class SymmetryGpaw1():
                def __init__(self, symop):
                    # self.op_scc = symop.rotation[None, :, :]
                    # self.cell_cv = symop.real_lattice
                    self.op_scc = symop.rotation_cart[None, :, :]
                    self.cell_cv = np.eye(3)
            symmetry = SymmetryGpaw1(self)
            R_aii = AtomRotations(setups.setups, setups.id_a, symmetry).get_R_asii()
            self.R_aii = [R_ii[0] for R_ii in R_aii]  # remove unnecessary nesting

    # def get_U_aii_gpaw(self, kpoint):
    #     """Phase corrected rotation matrices for the PAW projections."""
    #     return [ R_ii.T * np.exp( 2j * np.pi * np.dot(kpoiMax errnt, self.atom_map_T[a])) for a, R_ii in enumerate(self.R_aii)]
    #     # return [ R_ii.T  for a, R_ii in enumerate(self.R_aii)]  # no phase factor, try this

    def set_gpaw(self, calculator):
        """Set all gpaw-related attributes."""
        positions = calculator.atoms.get_scaled_positions()
        self.atom_map, self.atom_map_T = get_atom_map(self, positions)
        self.set_R_aii_gpaw(calculator)



    def rotate_projection(self, projections, k_origin, k_target):
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

        for a, R_ii in enumerate(self.R_aii):
            Pout_ni = (projections[a] @ R_ii.T)  # * np.exp(2j * np.pi * k_target @ self.atom_map_T[a])
            if self.time_reversal:
                Pout_ni = np.conj(Pout_ni)
            Pout_ni = Pout_ni * np.exp(2j * np.pi * k_target @ self.atom_map_T[a])
            I1, I2 = mapped_projections.map[self.atom_map[a]]
            mapped_projections.array[..., I1:I2] = Pout_ni
        return mapped_projections

    def rotate_pseudo_wavefunction(self, psi_n_grid, k_origin, k_target):
        """
        Rotate the pseudo wavefunction according to the given symmetry operation

        Parameters
        ----------
        psi_nG : np.ndarray(shape=(NB, n1, n2, n3), dtype=complex)
            the pseudo wavefunction in G-space
        k_origin : np.ndarray(shape=(3,), dtype=float)
            the original k-point in the basis of the reciprocal lattice
        k_target : np.ndarray(shape=(3,), dtype=float)
            the target k-point in the basis of the reciprocal lattice

        Returns
        -------
        psi_nG_rot : np.ndarray(shape=(NB, NG), dtype=complex)
            the rotated pseudo wavefunction in G-space
        """
        phase = np.exp(-2j * np.pi * self.transform_k(k_origin)  @ self.translation)

        psi_n_grid = np.array(psi_n_grid).copy()
        Nc = psi_n_grid.shape[1:]
        # Nc_tot = np.prod(Nc)
        if not self.is_identity:
            psi_n_grid = self.transform_grid_data(psi_n_grid)

        if self.time_reversal:
            psi_n_grid = np.conj(psi_n_grid)
        psi_n_grid *= phase  # should it be before or after time-reversal? TODO: check!
        kpt_shift = k_target - self.transform_k(k_origin)
        kpt_shift_int = np.round(kpt_shift).astype(int)
        assert np.allclose(kpt_shift, kpt_shift_int), f"k-point shift {kpt_shift} is not a reciprocal lattice vector"
        for i, ksh in enumerate(kpt_shift_int):
            if ksh != 0:
                phase = np.exp(-2j * np.pi * ksh * np.arange(Nc[i]) / Nc[i]).reshape((1,) * (i + 1) + (Nc[i],) + (1,) * (2 - i))
                psi_n_grid = psi_n_grid * phase
        return psi_n_grid







def get_atom_map(symop, positions):
    positions_list = UniqueListMod1(positions, tol=1e-4)
    assert len(positions) == len(UniqueListMod1(positions)), f"some positions are equivalent mod 1: {positions}"
    num_points = len(positions)
    atommap = -np.ones((num_points), dtype=int)
    T = np.zeros((num_points, 3), dtype=float)

    for ip, p in enumerate(positions):
        p2 = symop.transform_r(p)
        ip2 = positions_list.index(p2)
        atommap[ip] = ip2
        p2a = positions[ip2]
        T[ip] = p2a - p2
    assert np.all(atommap >= 0), f"some positions are not mapped: atommap={atommap}"

    T_round = np.round(T)
    T_diff = np.abs(T - T_round).max()
    if T_diff > 1e-6:
        msg = f"the T vectors should result integer values, but the maximal deviation from integer is {T_diff} T=\n{T}, \nT_round=\n{T_round}, \n max_diff={T_diff}"
        if T_diff > 1e-3:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    T = T_round.astype(int)
    return atommap, T
