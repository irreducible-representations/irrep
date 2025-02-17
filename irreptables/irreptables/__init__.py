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

__version__="1.1.1"

import os
import logging
import numpy as np
from irrep.utility import str2bool, str2list_space, str_, log_message
from functools import cached_property

# using a logger to print useful information during debugging,
# set to logging.INFO to disable debug messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Matrices of generators of point groups in the convencional setting
# Used when SpaceGroup_SVD is called with mode='create'

E = np.eye(3)
I = - E
C2z = np.array([[-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]])
C3z = np.array([[0, -1, 0],
                [1, -1, 0],
                [0, 0, 1]])
C4z = np.array([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])
C6z = np.array([[1, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])
C2y = np.array([[-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]])
C3111 = np.array([[0, 0, 1],
                 [1, 0, 0],
                 [0, 1, 0]])
C110 = np.array([[0, 1, 0],  # in trigonal/hexagonal
                 [1, 0, 0],
                 [0, 0, -1]])
C1m10 = np.array([[0, -1, 0],  # in trigonal/hexagonal
                 [-1, 0, 0],
                 [0, 0, -1]])

My = I @ C2y
M110 = I @ C110  # in trigonal/hexagonal
M1m10 = I @ C1m10  # in trigonal/hexagonal
mC4z = I @ C4z
mC6z = I @ C6z


generators = {}
generators['C1'] = np.array([E])  # identity gr.
generators['Ci'] = np.array([I])  # inversion gr.

# For monoclonic groups
generators['Cs'] = np.array([My])  # reflection gr. (b-axis for monoclinic)
generators['C2'] = np.array([C2y])  # 2-fold gr. (b-axis for monoclinic)
generators['C2h'] = np.array([C2y, I])  # inversion gr.

# For orthorhombic groups
generators['D2'] = np.array([C2z, C2y])  # 222
generators['C2v'] = np.array([C2z, My])  # mm2
generators['D2h'] = np.array([C2z, C2y, I])  # mmm

# For tetragonal groups
generators['C4'] = np.array([C4z])  # 4-fold gr.
generators['S4'] = np.array([mC4z])  # -4 gr.
generators['C4h'] = np.array([C4z, I])  # 4/m
generators['D4'] = np.array([C4z, C2y])  # 422
generators['C4v'] = np.array([C4z, My])  # 4mm
generators['D4h'] = np.array([C4z, C2y, I])  # 4/mmm
generators['D2d(1)'] = np.array([mC4z, C2y])  # -42m (2-fold along cell vecs)
generators['D2d(2)'] = np.array([mC4z, My])  # -4m2 (mirrors perpendicular to cell vecs)

# For trigonal groups
generators['C3'] = np.array([C3z])  # 3-fold gr.
generators['S6'] = np.array([C3z, I])  # -3 gr.
generators['D3(1)'] = np.array([C3z, C1m10])  # 312
generators['D3(2)'] = np.array([C3z, C110])  # 321
generators['C3v(1)'] = np.array([C3z, M110])  # 3m1
generators['C3v(2)'] = np.array([C3z, M1m10])  # 31m
generators['D3d(1)'] = np.array([C3z, C1m10, I])  # -31m
generators['D3d(2)'] = np.array([C3z, C110, I])  # -3m1

# For hexagonal groups
generators['C6'] = np.array([C6z])  # 6-fold gr.
generators['C3h'] = np.array([mC6z])  # -6 gr.
generators['C6h'] = np.array([C6z, I])  # 6/m
generators['D6'] = np.array([C6z, C110])  # 622
generators['C6v'] = np.array([C6z, M110])  # 6mm
generators['D6h'] = np.array([C6z, C110, I])  # 6/mmm
generators['D3h(1)'] = np.array([mC6z, C110])  # -62m (2-fold along cell vecs)
generators['D3h(2)'] = np.array([mC6z, M110])  #  -6m2 (mirrors perpendicular to cell vecs)

# For cubic groups
generators['T'] = np.array([C2z, C2y, C3111])  # 23
generators['Th'] = np.array([C2z, C2y, C3111, I])  # m-3
generators['O'] = np.array([C4z, C3111])  # 432
generators['Td'] = np.array([mC4z, C3111, C110])  # -432
generators['Oh'] = np.array([C4z, C3111, I])  # m-3m


class SymopTable:
    '''
    Parses a `str` that  describes a symmetry operation of the space-group and 
    stores info about it in attributes.

    Parameters
    ----------
    line : str
        Line to be parsed, which describes a symmetry operation.

    Attributes
    ----------
    R : array, shape=(3,3)
        Rotational part, describing the transformation of basis vectors (not 
        cartesian coordinates!).
    t : array, shape=(3,)
        Direct coordinates of the translation vector.
    S : array, shape=(2,2)
        SU(2) matrix describing the transformation of spinor components.
    '''

    def __init__(self, line):
        numbers = line.split()
        self.R = np.array(numbers[:9], dtype=int).reshape(3, 3)
        self.t = np.array(numbers[9:12], dtype=float)
        if len(numbers) > 12:
            self.S = (
                np.array(numbers[12:16], dtype=float)
                * np.exp(1j * np.pi * np.array(numbers[16:20], dtype=float))
            ).reshape(2, 2)
        else:
            self.S = np.eye(2)

    def str(self, spinor=True):
        """
        Create a `str` describing the symmetry operation as implemented in the 
        files included in `IrRep`.

        Parameters
        ----------
        spinor : bool, default=True
            `True` if the matrix describing the transformation of spinor 
            components should be written.

        Returns
        -------
        str
        """
        return (
            "   ".join(" ".join(str(x) for x in r) for r in self.R)
            + "     "
            + " ".join(str_(x) for x in self.t)
            + (
                (
                    "      "
                    + "    ".join(
                        "  ".join(str_(x) for x in X)
                        for X in (
                            np.abs(self.S.reshape(-1)),
                            np.angle(self.S.reshape(-1)) / np.pi,
                        )
                    )
                )
                if spinor
                else ""
            )
        )


class KPoint:
    """
    Organizes the info about a maximal k-point and contains routines to print 
    it. This info is obtained by parsing the parameter `line` or passed 
    directly as `name`, `k` and `isym`.
    
    Parameters
    ----------
    name : str, default=None
        Label of the k-point.
    k : array, default=None
        Direct coordinates of the k-point.
    isym : array, default=None
        Indices of symmetry operations in the little co-group. Indices make 
        reference to the symmetry operations stored in the header of the file 
        and stored in `IrrepTable.symmetries`. 
    line : str, default=None
        Line to be parsed. 
    
    Attributes
    ----------
    name : str
        Label of the k-point. 
    k : array, shape=(3,) 
        Direct coordinates of the k-point.
    isym : array
        Indices of symmetry operations in the little co-group. Indices make 
        reference to the symmetry operations stored in the header of the file 
        and stored in `IrrepTable.symmetries`. 
    """

    def __init__(self, name=None, k=None, isym=None, line=None):

        if line is not None:
            line_ = line.split(":")
            if line_[0].split()[0] != "kpoint":
                raise ValueError
            self.name = line_[0].split()[1]
            self.k = np.array(line_[1].split(), dtype=float)
            self.isym = str2list_space(
                line_[2]
            )  # [ int(x) for x in line_[2].split() ]  #
        else:
            self.name = name
            self.k = k
            self.isym = isym

    def __eq__(self, other):
        """
        Compares the attributes of this class with those of class instance 
        `other`.

        Parameters
        ----------
        other : class
            Instance of class `KPoint`.

        Returns
        -------
        bool
            `True` if all attributes have identical value, `False` otherwise.
        """
        if self.name != other.name:
            return False
        if np.linalg.norm(self.k - other.k) > 1e-8:
            return False
        if self.isym != other.isym:
            return False
        return True

    def show(self):
        """
        Create a `str` containing the values of all attributes.

        Returns
        -------
        str
            Line showing the values of all attributes.
        """
        return "{0} : {1}  symmetries : {2}".format(self.name, self.k, self.isym)

    def str(self):
        '''
        Create a `str` containing the values of all attributes.

        Returns
        -------
        str
            Line that, when parsed, would lead to an instance of class `KPoint` 
            with identical values of attributes.
        '''
        return "{0} : {1}  : {2}".format(
            self.name,
            " ".join(str(x) for x in self.k),
            " ".join(str(x) for x in sorted(self.isym)),
        )


class Irrep:
    """
    Parses the line containing the description of the irrep and stores the info 
    in its attributes. Contains methods print descriptions of the irrep. 

    Parameters
    ----------
    f : file object, default=None 
        It corresponds to the file containing the info about the space-group 
        and its irreps.
    nsym_group : int, default=None
        Number of symmetry operations in the "point-group" of the space-group.
    line : str, default=None
        Line with the description of an irrep, read from the file containing 
        info about the space-group and irreps.
    k_point : class instance, default=None
        Instance of class `KPoint`.
    v : int, default=0
        Verbosity level. Default set to minimalistic printing

    Attributes
    ----------
    k : array, shape=(3,) 
        Direct coordinates of a k-point.
    kpname : str
        It is the label of a k-point.
    name : str
        Label of the irrep.
    dim : int
        Dimension of the irrep.
    nsym : int
        Number of symmetry operations in the little co-group of the k-point.
    reality : bool
        `True` if traces of all symmetry operations are real, `False` 
        otherwise.
    characters : dict
        Each key is the index of a symmetry operation in the little co-group 
        and the corresponding value is the trace of that symmetry in the irrep.
    """

    def __init__(self, line, k_point, v=0):
        logger.debug("reading irrep line <{0}> for KP=<{1}> ".format(line, k_point.str()))
        self.k = k_point.k
        self.kpname = k_point.name
        line = line.split()
        self.name = line[0]
        self.dim = int(line[1])
        self.nsym = len(k_point.isym)
        self.reality = len(line[2:]) == self.nsym
        ch = np.array(line[2 : 2 + self.nsym], dtype=float)
        if not self.reality:
            ch = ch * np.exp(
                1.0j
                * np.pi
                * np.array(line[2 + self.nsym : 2 + 2 * self.nsym], dtype=float)
            )
        self.characters = {k_point.isym[i]: ch[i] for i in range(self.nsym)}
        log_message(f"## Irrep {self.name}\nCharacter:\n{self.characters}", v, 2)
        assert len(self.characters) == self.nsym

    def show(self):
        """
        Print label of the k-point and info about the irrep.
        """
        print(self.kpname, self.name, self.dim, self.reality)

    def str(self):
        """
        Generate a line describing the irrep and its character.

        Returns
        -------
        str
            Line describing the irrep, as it is written in the table of the
            space-group included in `IrRep`. This line contains the label, 
            dimension and character of the irrep.
        """
        logger.debug(self.characters)
        ch = np.array([self.characters[isym] for isym in sorted(self.characters)])
        if np.abs(np.imag(ch)).max() > 1e-6:
            str_ch = "   " + "  ".join(str_(x) for x in np.abs(ch))
            str_ch += "   " + "  ".join(str_(x) for x in np.angle(ch) / np.pi)
        else:
            str_ch = "   " + "  ".join(str_(x) for x in np.real(ch))
        return self.name + " {} ".format(self.dim) + str_ch


class IrrepTable:
    """
    Parse file corresponding to a space-group, storing the info in attributes. 
    Also contains methods to print and write this info in a file.

    Parameters
    ----------
    SGnumber : int
        Number of the space-group.
    spinor : bool
        `True` if the matrix describing the transformation of spinor components 
        should be read.
    name : str, default=None
        Name of the file from which info about the space-group and irreps 
        should be read. If `None`, the code will try to open a file already 
        included in it.
    v : int, default=0
        Verbosity level. Default set to minimalistic printing

    Attributes
    ----------
    number : int
        Number of the space-group.
    name : str
        Symbol of the space-group in Hermann-Mauguin notation. 
    spinor : bool
        `True` if wave-functions are spinors (SOC), `False` if they are scalars.
    nsym : int
       Number of symmetry operations in the "point-group" of the space-group. 
    symmetries : list
        Each component is an instance of class `SymopTable` corresponding to a 
        symmetry operation in the "point-group" of the space-group.
    NK : int
        Number of maximal k-points in the Brillouin zone.
    irreps : list
        Each component is an instance of class `IrRep` corresponding to an 
        irrep of the little group of a maximal k-point.
    kpoints : list
        Each element is an instance of class `KPoint` for a k point
    """

    def __init__(self, SGnumber, spinor, name=None, v=0):
        self.number = SGnumber
        self.spinor = spinor
        if name is None:
            name = "{root}/tables/irreps-SG={SG}-{spinor}.dat".format(
                SG=self.number,
                spinor="spin" if self.spinor else "scal",
                root=os.path.dirname(__file__),
            )
            msg = f"Reading standard irrep table <{name}>"
            log_message(msg, v, 2)
        else:
            msg = f"Reading a user-defined irrep table <{name}>"
            log_message(msg, v, 2)

        log_message("\n---------- DATA FROM THE TABLE ----------\n", v, 2)
        lines = open(name).readlines()[-1::-1]
        while len(lines) > 0:
            l = lines.pop().strip().split("=")
            # logger.debug(l,l[0].lower())
            if l[0].lower() == "SG":
                assert int(l[1]) == self.number
            elif l[0].lower() == "name":
                self.name = l[1]
            elif l[0].lower() == "nsym":
                self.nsym = int(l[1])
            elif l[0].lower() == "spinor":
                assert str2bool(l[1]) == self.spinor
            elif l[0].lower() == "symmetries":
                log_message("Reading symmetries from tables", v, 2)
                self.symmetries = []
                while len(self.symmetries) < self.nsym:
                    l = lines.pop()
                    # logger.debug(l)
                    try:
                        self.symmetries.append(SymopTable(l))
                    except Exception as err:
                        logger.debug(err)
                        pass
                break

        msg = "Symmetries are:\n" + "\n".join(s.str() for s in self.symmetries)
        log_message(msg, v, 2)

        self.irreps = []
        self.kpoints = []
        while len(lines) > 0:
            l = lines.pop().strip()
            try:
                kp = KPoint(line=l)
                self.kpoints.append(kp)
                msg = f"k-point successfully read:\n{kp.str()}"
                log_message(msg, v, 2)
            except Exception as err:
                try:
                    self.irreps.append(Irrep(line=l, k_point=kp, v=v))
                except Exception as err:
                    if len(l.split()) > 0:
                        msg = ("WARNING: could not parse k-point nor irrep from the "
                               "following line <\n{}>"
                               .format(l))
                        log_message(msg, v, 2)
                    else:
                        pass

    def show(self):
        '''
        Print info about symmetry operations and irreps.  
        '''
        for i, s in enumerate(self.symmetries):
            print(i + 1, "\n", s.R, "\n", s.t, "\n", s.S, "\n\n")
        for irr in self.irreps:
            irr.show()


class SpaceGroup_SVD:
    '''
    Class used to determine the transformation from the primitive cell 
    to the conventional cell of tables.

    Attributes
    ----------
    number : int
        Number of the space group
    mode : str
        Whether class has to be created by parsing tables or not. The 
        later is typically the case if you want to redo the SVD.
    rotations : array
        First index labels the generators of the point group, and the 
        corresponding value is its matrix in the primitive cell, based on
        the conventional transformation matrix to the primitive cell
    translations : array
        Each row is the translations vector of a generator in the 
        standard-primitive cell
    num_gens : int
        Number of generators of the point group.
    file : str
        Name of the data file of the space group.
    centering : str
        Letter identifying the centering of the space group in the tables
    to_primitive : array
        Transformation from conventional cell to the standard-primitive cell
        (same as in `vasp2trace`).
    N_matrix : array, shape=(num_gens*3,3)
        Matrices of generators in the standard-primitive cell stacked 
        vertically
    lambda_matrix : array
        Matrix that has to be multiplied to the differences of translational 
        parts
    '''

    def __init__(self, sg_number, mode='create'):
        '''
        Parameters
        ----------
        sg_number : int
            Number of the space group
        mode : str, default='create'
            Pass the value 'parse' to create the instance by parsing 
            the data from tables. Pass 'create' if you are modifying 
            the data in the tables.
        '''

        self.number = sg_number
        self.mode = mode
        self.file = '{}/svd_data/svd-{}.dat'.format(
                        os.path.dirname(__file__),
                        self.number)
        self.rotations, self.translations = self.get_generators()  # in primitive cell
        self.num_gens = len(self.rotations)

    def get_generators(self):

        if self.mode == 'create':

            matrices = generators[self.point_group]

            # Identify generators in tables to get translational parts
            table = IrrepTable(self.number, spinor=False)
            inds_generators = []
            for isvd, W_svd in enumerate(matrices):
                found = False
                for i, sym_table in enumerate(table.symmetries):
                    if np.allclose(sym_table.R, W_svd):
                        inds_generators.append(i)
                        found = True
                        break
                if not found:
                    print(f'{isvd} not matched!')

            translations = []
            for i in inds_generators:
                translations.append(table.symmetries[i].t)
            translations = np.array(translations)

            # Generators in primitive cell
            matrices = np.einsum('ja,iab,bk',
                             np.linalg.inv(self.to_primitive),
                             matrices,
                             self.to_primitive)
            translations = np.einsum('ij,kj->ki',
                             np.linalg.inv(self.to_primitive),
                             translations)

        elif self.mode == 'parse':  # Parse from data file
            f = open(self.file, 'r')
            num_gens = int(f.readline().split()[1])
            matrices = np.zeros((num_gens, 3, 3), dtype=int)
            translations = np.zeros((num_gens, 3), dtype=float)
            for i in range(num_gens):
                line = f.readline().split()
                matrices[i] = np.reshape(line[:9], newshape=(3,3))
                translations[i] = np.array(line[9:], dtype=float)
            f.close()

        # Check that matrices of generators are integers
        diff = matrices - np.array(matrices, dtype=int)
        diff = np.max(np.abs(diff))
        if diff > 1e-5:
            print('WARNING: matrices should be integers in primitive basis. '
                  'Found a difference of {} w.r.t. integers'
                  .format(diff))

        return matrices, translations

    @cached_property
    def N_matrix(self):

        N = self.rotations.reshape(self.num_gens*3,3)
        return N

    def svd(self):
        U, S, V = np.linalg.svd(self.N_matrix)
        return U, S, V

    @cached_property
    def lambda_matrix(self):

        if self.mode == 'create':
            U, S, V = self.svd()
            S_matrix = np.zeros(self.N_matrix.shape, dtype=float)
            S_matrix[:len(S)] = np.diag(1.0/S)
            T = S_matrix.T
            Lambda = V @  T @ U

        elif self.mode == 'parse':
            Lambda = np.zeros((3, 3*self.num_gens), dtype=float)
            f = open(self.file, 'r')
            for i in range(self.num_gens + 1):
                f.readline()
            for i in range(3):
                Lambda[i] = np.array(f.readline().split())
            f.close()

        return Lambda

    def save_file(self):

        print(f'Saving data into --> {self.file}')
        print('WARNING: this file will be overwritten.')
        f = open(self.file, 'w')
        f.write(f'{self.number}  {self.num_gens}\n')
        for R, t in zip(self.rotations, self.translations):
            R = R.reshape(9)
            s = [f'{int(x):2d}' for x in R]
            s += [f'{x:5.2f}' for x in t]
            s = '  '.join(s)
            f.write(s)
            f.write('\n')
        for row in self.lambda_matrix:
            s = [f'{x:10.6f}' for x in row]
            s = '  '.join(s)
            f.write(s)
            f.write('\n')
        f.close()

    @property
    def centering(self):

        if self.number in (5,8,9,12,15,20,21,35,36,37,63,64,65,66,67,68):
            return 'C'
        elif self.number in (38,39,40,41):
            return 'A'
        elif self.number in (22,42,43,69,70,196,216,226,202,227,203,228,209,219,210,225):
            return 'F'
        elif self.number in (46,71,121,72,82,87,97,107,122,23,73,88,98,24,44,74,79,109,
                             119,139,45,80,110,120,140,141,206,211,142,197,217,199,214,229,220,230):
            return 'I'
        elif self.number in (146, 148, 155, 160, 161, 166, 167):
            return 'R'
        else:
            return 'P'

    @cached_property
    def to_primitive(self):

        if self.centering  == 'P':
            return np.eye(3)
        elif self.centering == 'C':
            return np.array([[0.5, 0.5, 0.0],
                             [-0.5, 0.5, 0.0],
                             [0.0, 0.0, 1.0]])
        elif self.centering == 'A':
            return np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.5, -0.5],
                             [0.0, 0.5, 0.5]])
        elif self.centering == 'F':
            return np.array([[0.0, 0.5, 0.5],
                             [0.5, 0.0, 0.5],
                             [0.5, 0.5, 0.0]])
        elif self.centering == 'I':
            return np.array([[-0.5, 0.5, 0.5],
                             [0.5, -0.5, 0.5],
                             [0.5, 0.5, -0.5]])
        elif self.centering == 'R':
            return np.array([[2./3., -1./3., -1./3.],
                             [1./3., 1./3., -2./3.],
                             [1./3., 1./3., 1./3.]])

    @cached_property
    def point_group(self):

        if self.number == 1:
            return 'C1'
        elif self.number == 2:
            return 'Ci'
        elif self.number in (3, 4, 5):
            return 'C2'
        elif self.number in (6, 7, 8, 9):
            return 'Cs'
        elif self.number in np.arange(10, 16):
            return 'C2h'
        elif self.number in np.arange(16, 25):
            return 'D2'
        elif self.number in np.arange(25, 47):
            return 'C2v'
        elif self.number in np.arange(47, 75):
            return 'D2h'
        elif self.number in np.arange(75, 81):
            return 'C4'
        elif self.number in np.arange(81, 83):
            return 'S4'
        elif self.number in np.arange(83, 89):
            return 'C4h'
        elif self.number in np.arange(89, 99):
            return 'D4'
        elif self.number in np.arange(99, 111):
            return 'C4v'
        elif self.number in (111,112,113,114,121,122):
            return 'D2d(1)'
        elif self.number in (115,116,117,118,119,120):
            return 'D2d(2)'
        elif self.number in np.arange(123, 143):
            return 'D4h'
        elif self.number in np.arange(143, 147):
            return 'C3'
        elif self.number in np.arange(147, 149):
            return 'S6'
        elif self.number in (149, 151, 153):
            return 'D3(1)'
        elif self.number in (150, 152, 154, 155):
            return 'D3(2)'
        elif self.number in (157, 159):
            return 'C3v(2)'
        elif self.number in (156, 158, 160, 161):
            return 'C3v(1)'
        elif self.number in (162, 163):
            return 'D3d(1)'
        elif self.number in (164, 165, 166, 167):
            return 'D3d(2)'
        elif self.number in np.arange(168, 174):
            return 'C6'
        elif self.number == 174:
            return 'C3h'
        elif self.number in (175, 176):
            return 'C6h'
        elif self.number in np.arange(177, 183):
            return 'D6'
        elif self.number in np.arange(183, 187):
            return 'C6v'
        elif self.number in (187, 188):
            return 'D3h(2)'
        elif self.number in np.arange(189, 191):
            return 'D3h(1)'
        elif self.number in np.arange(191, 195):
            return 'D6h'
        elif self.number in np.arange(195, 200):
            return 'T'
        elif self.number in np.arange(200, 207):
            return 'Th'
        elif self.number in np.arange(207, 215):
            return 'O'
        elif self.number in np.arange(215, 221):
            return 'Td'
        elif self.number in np.arange(221, 231):
            return 'Oh'
