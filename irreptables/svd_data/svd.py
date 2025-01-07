import numpy as np
from functools import cached_property

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
    generators : array
        First index labels the generators of the point group, and the 
        corresponding value is its matrix in the primitive cell, based on
        the conventional transformation matrix to the primitive cell
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
        self.file = f'svd-{self.number}.dat' 
        self.generators = self.get_generators()  # in primitive cell
        self.num_gens = len(self.generators)

    def get_generators(self):

        if self.mode == 'create':
            matrices = generators[self.point_group]

        elif self.mode == 'parse':  # Parse from data file
            f = open(self.file, 'r')
            num_gens = int(f.readline().split()[1])
            matrices = np.zeros((num_gens, 3, 3), dtype=int)
            for i in range(num_gens):
                matrices[i] = np.reshape(f.readline().split(), shape=(3,3))
            f.close()

        # Matrices of generators in primitive cell
        matrices = np.einsum('ja,iab,bk', 
                             np.linalg.inv(self.to_primitive),
                             matrices,
                             self.to_primitive)
        
        # Check that matrices of generators are integers
        diff = matrices - np.array(matrices, dtype=int)
        diff = np.max(np.abs(diff))
        if diff > 1e-5:
            print('WARNING: matrices should be integers in primitive basis. '
                  'Found a difference of {} w.r.t. integers'
                  .format(diff))

        return matrices
            
    @cached_property
    def N_matrix(self):

        N = self.generators.reshape(self.num_gens*3,3)
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
        for matrix in self.generators:
            matrix = matrix.reshape(9)
            s = [f'{int(x):2d}' for x in matrix]
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
