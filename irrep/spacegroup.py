
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


from functools import cached_property
import warnings
import numpy as np
import spglib

from irrep.readfiles import ParserAbinit, ParserEspresso, ParserGPAW, ParserVasp, ParserW90

from .symmetry_operation import SymmetryOperation
from .utility import BOHR, log_message
from packaging import version
import os


class SpaceGroup:

    """
    Class to represent a space-group. Contains methods to describe and print
    info about the space-group.

    Parameters
    ----------
    Lattice : array, shape=(3,3)
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space. in Angstroms.
    spinor : bool, default=True
        `True` if wave-functions are spinors (SOC), `False` if they are scalars.
    rotations : list
        Each element is a 3x3 array describing the rotation matrix of a 
        symmetry operation.
    translations : list
        Each element is a 3-element array describing the translational part 
        of a symmetry operation.
    time_reversals : list
        Each element is a boolean indicating whether the symmetry operation 
        is time-reversal or not.
    number : int, default=0
        Number of the space-group. If not specified, it is set to 0.
    name : str, default=""
        Symbol of the space-group in Hermann-Mauguin notation. If not 
        specified, it is set to an empty string.
    spinor_rotations : list, default=None
        Each element is a 2x2 array describing the transformation of the 
        spinor under the symmetry operation. If not specified, it is set to 
        `None`, which means that no spinor rotations are considered.
    symemtry_operations : list, default=None
        Each element is an instance of class `SymmetryOperation` corresponding 
        to a unitary symmetry in the point group of the space-group. If provided,
        it overrides the `rotations`, `translations`, `time_reversals` and
        `spinor_rotations` parameters.
    copy_symops : bool, default=False
        If `True`, the symmetry operations are copied to avoid modifying the 
        original ones. If `False`, the original symmetry operations are used.
        (is symmetry_operations is None, this parameter is ignored)
    translation_mod1 : bool, default=True
        If `True`, the translational part of the symmetry operations is taken modulo 1,
        otherwise it is taken as is. Thi is useful to match with specific consventions (e.g. when reading from the .sym file).

    Attributes
    ----------
    symmetries : list
        Each element is an instance of class `SymmetryOperation` corresponding 
        to a unitary symmetry in the point group of the space-group.
    name : str
        Symbol of the space-group in Hermann-Mauguin notation.
    number_str : str
        String representation of the number of the space-group. 
    spinor : bool
        `True` if wave-functions are spinors (SOC), `False` if they are scalars.
    au_symmetries : list
        Each element is an instance of class `SymmetryOperation` corresponding 
        to an antiunitary symmetry in the point group of the space-group.
    name : str 
        Symbol of the space-group in Hermann-Mauguin notation. 
    number : int 
        Number of the space-group.
    Lattice : array, shape=(3,3) 
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    positions : array
        Direct coordinate of sites in the DFT cell setting.
    typat : list
        Indices to identify the element in each atom. Atoms of the same element 
        share the same index.
    refUC : array, default=None
        3x3 array describing the transformation of vectors defining the 
        unit cell to the standard setting.
    shiftUC : array, default=None
        Translation taking the origin of the unit cell used in the DFT 
        calculation to that of the standard setting.
    alat : float (optional)
        Lattice parameter in angstroms (quantum espresso convention).
    magnetic : bool

    Notes
    -----
    The symmetry operations to which elements in the attribute `symmetries` 
    correspond are the operations belonging to the point group of the 
    space-group :math:`G`, i.e. the coset representations :math:`g_i` 
    taking part in the coset decomposition of :math:`G` w.r.t the translation 
    subgroup :math:`T`:
    ..math:: G=T+ g_1 T + g_2 T +...+ g_N T 
    """

    def __init__(self, Lattice, spinor,
                 rotations=None, translations=None,
                 time_reversals=None, translation_mod1=False,
                 symemtry_operations=None,
                 number=None,
                 number_str=None,
                 name="",
                 spinor_rotations=None,
                 copy_symops=False,
                 typat=None,
                 positions=None,
                 magnetic=None,
                 refUC=None,
                 shiftUC=None,
                 alat=None,
                 verbosity=0,
                 ):

        log_message(f"Creating space group {name} with number {number_str}", verbosity, 2)
        log_message(f"Number of symmetries: {len(rotations)}", verbosity, 2)

        self.real_lattice = Lattice
        self.spinor = spinor
        self.name = name
        self.typat = typat
        self.positions = positions
        self.magnetic = magnetic
        self.refUC = refUC
        self.shiftUC = shiftUC
        self.alat = alat

        if number_str is not None:
            self.number_str = number_str
            assert number is None, "number and number_str cannot be set at the same time"
        elif number is not None:
            try:
                number = int(number)
            except ValueError:
                raise ValueError(f"number must be an integer (or convertable to int), got <{number}> ({type(number)}) ")

            if number < 0:
                warnings.warn("Negative space group number is not supported. Setting it to -1")
                number = -1
            self.number_str = str(number)
        else:
            self.numper_str = "0"

        if symemtry_operations is None:
            copy_symops = False
            symmetry_operations = []
            if spinor_rotations is None:
                spinor_rotations = [None] * len(rotations)

            for i, (rot, trans, tr, srot) in enumerate(zip(rotations,
                                                        translations,
                                                        time_reversals,
                                                        spinor_rotations)):
                symmetry_operations.append(SymmetryOperation(rot=rot,
                                                        trans=trans,
                                                        ind=i + 1,
                                                        Lattice=self.real_lattice,
                                                        time_reversal=tr,
                                                        spinor=self.spinor,
                                                        translation_mod1=translation_mod1,
                                                        spinor_rotation=srot))
        if copy_symops:
            # Copy symmetries to avoid modifying the original ones
            self.symmetries = [s.copy() for s in symmetry_operations]
        else:
            self.symmetries = symmetry_operations


    def as_dict(self):
        """
        return dictionary with info essential about the spacegroup
        """
        return dict(
            Lattice=self.real_lattice,
            spinor=self.spinor,
            rotations=[s.rotation for s in self.symmetries],
            translations=[s.translation for s in self.symmetries],
            spinor_rotations=[s.spinor_rotation for s in self.symmetries],
            time_reversals=[s.time_reversal for s in self.symmetries],
            number=self.number if self.number is not None else -1,
            name=self.name if self.name is not None else "unknown"
        )

    @property
    def size(self):
        """
        Number of symmetry operations in the space-group.
        """
        return len(self.symmetries)

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

        # Print cell vectors in DFT cell only
        print('Cell vectors in angstroms:\n')
        print(f"{'Vectors of DFT cell':^32}")
        for i in range(3):
            vec1 = self.real_lattice[i]
            print(f'a{i:d} = {vec1[0]:7.4f}  {vec1[1]:7.4f}  {vec1[2]:7.4f}  ')
        print()

        print()
        print('\n ---------- SPACE GROUP ----------- \n')
        print()
        print(f'Space group: {self.name} (# {self.number_str})')
        print(f'Number of symmetries: {self.size} (mod. lattice translations)')
        for symop in self.symmetries:
            if symmetries is None or symop.ind in symmetries:
                symop.show(refUC=None, shiftUC=None)

    @property
    def lattice(self):
        return self.real_lattice

    @property
    def Lattice(self):
        return self.real_lattice


    @cached_property
    def lattice_inv(self):
        return np.linalg.inv(self.lattice)

    @cached_property
    def reciprocal_lattice(self):
        return self.lattice_inv.T * (2 * np.pi)


    @classmethod
    def from_cell(
            CLS,
            cell=None,
            real_lattice=None,
            positions=None,
            typat=None,
            spinor=True,
            alat=None,
            from_sym_file=None,
            magmom=None,
            include_TR=True,
            symprec=1e-5,
            mag_symprec=-1,
            angle_tolerance=-1,
            verbosity=0,
            ############
            **kwargs_tables
    ):
        """
        Determine the space-group and save info in attributes. Contains methods to 
        describe and print info about the space-group.

        Parameters
        ----------
        cell : tuple
            Override the `real_lattice`, `positions`, `typat` and `magmom`
            `cell[0]` is a 3x3 array where cartesian coordinates of basis 
            vectors **a**, **b** and **c** are given in rows. 
            `cell[1]` is an array
            where each row contains the direct coordinates of an ion's position. 
            `cell[2]` is an array where each element is a number identifying the 
            atomic species of an ion.
            `cell[3]` (optional) is an array where each element is the magnetic
            moment of an ion.
            See `get_symmetry` in 
            `Spglib <https://spglib.github.io/spglib/python-spglib.html#get-symmetry>`_.
        real_lattice : array( (3, 3), dtype=float)
            3x3 array with cartesian coordinates of basis row-vectors forming the
                unit-cell in real space. 
        positions : array, shape=(Nions, 3), dtype=float
            Each row contains the fractional coordinates of an atom in the unit cell.
        typat : array, shape=(Nions,), dtype=int
            Each element is an integer identifying the atomic species of an ion.
            Atoms of the same element share the same index.
        spinor : bool, default=True
            `True` if wave-functions are spinors (SOC), `False` if they are scalars.
        refUC : array, default=None
            3x3 array describing the transformation of vectors defining the 
            unit cell to the standard setting.
        shiftUC : array, default=None
            Translation taking the origin of the unit cell used in the DFT 
            calculation to that of the standard setting.
        search_cell : bool, default=False
            Whether the transformation to the conventional cell should be computed.
            It is `True` if kpnames was specified in CLI.
        trans_thresh : float, default=1e-5
            Threshold used to compare translational parts of symmetries.
        alat : float, default=None
            Lattice parameter in angstroms (quantum espresso convention).
        from_sym_file : str, default=None
            If provided, the symmetry operations are read from this file.
            (format of pw2wannier90 prefix.sym  file)
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing
        magmom : array
            Each element is the magnetic moment of an ion. if None - non-magnetic calculation
            if True - magnetic moments are set to zero, i.e. time-reversal symmetry is included in the spacegroup
        symprec, angle_tolerance, mag_symprec: float
            see `get_symmetry` and 'get_magnetic_symmetry` in 
            `Spglib documentation <https://spglib.readthedocs.io/en/stable/api/python-api.html#spglib.spglib.get_magnetic_symmetry>'__

        """


        if cell is not None:
            assert len(cell) in (3, 4), "cell must be a tuple of length 3"
            assert real_lattice is None, "real_lattice must NOT be provided if cell is given"
            real_lattice = np.array(cell[0])
            assert positions is None, "positions must NOT be provided if cell is given"
            positions = np.array(cell[1])
            assert typat is None, "typat must NOT be provided if cell is given"
            typat = cell[2]
            if len(cell) == 4:
                assert magmom is None, "magmom must NOT be provided if cell is given with 4 elements"
                magmom = cell[3]
        assert real_lattice is not None, "real_lattice must be provided"
        assert positions is not None, "positions must be provided"
        assert typat is not None, "typat must be provided"
        magmom = magmom
        include_TR = include_TR

        if not np.all(isinstance(x, int) for x in typat):
            log_message("typat are not integers -trying to enumerate them", verbosity, 2)
            typeatset = set(typat)
            typeatdict = {t: i + 1 for i, t in enumerate(typeatset)}
            typat = np.array([typeatdict[t] for t in typat], dtype=int)

        if magmom is not None or include_TR:
            magnetic = True
        else:
            magnetic = False

        cell = (real_lattice, positions, typat)
        if not magnetic:  # No magnetic moments magmom = None

            dataset = spglib.get_symmetry_dataset(cell,
                                                  symprec=symprec,
                                                  angle_tolerance=angle_tolerance)
            if version.parse(spglib.__version__) < version.parse('2.5.0'):
                name = dataset['international']
                number_str = str(dataset['number'])
                refUC = dataset['transformation_matrix']
                shiftUC = dataset['origin_shift']
                rotations = dataset['rotations']
                translations = dataset['translations']
            else:
                name = dataset.international
                number_str = str(dataset.number)
                refUC = dataset.transformation_matrix
                shiftUC = dataset.origin_shift
                rotations = dataset.rotations
                translations = dataset.translations
            time_reversal_list = [False] * len(rotations)  # to do: change it to implement grey groups

        else:  # Magnetic group
            if magmom is None or magmom is True:
                magmom = np.zeros((len(positions), 3), dtype=float)
            dataset = spglib.get_magnetic_symmetry_dataset((*cell, magmom),
                                                           symprec=symprec,
                                                           angle_tolerance=angle_tolerance,
                                                           mag_symprec=mag_symprec)
            if dataset is None:
                raise ValueError("No magnetic space group could be detected!")
            rotations = dataset.rotations
            translations = dataset.translations
            time_reversal_list = dataset.time_reversals
            refUC = dataset.transformation_matrix
            shiftUC = dataset.origin_shift

            uni_number = dataset.uni_number
            root = os.path.dirname(__file__)
            with open(root + "/data/msg_numbers.data", 'r') as f:
                number_str, name = f.readlines()[uni_number].strip().split(" ")

        # Read syms from .sym file (useful for Wannier interface)
        if from_sym_file is not None:
            assert alat is not None, "Lattice parameter must be provided to read symmetries from file"
            rot_cart, trans_cart = read_sym_file(from_sym_file)
            rotations, translations = cart_to_crystal(rot_cart,
                                                      trans_cart,
                                                      real_lattice,
                                                      alat)
            translation_mod1 = False
        else:
            translation_mod1 = True

        if not include_TR:
            log_message("Removing time-reversal symmetries", verbosity, 2)
            log_message(f"Number of time-reversal symmetries: {sum(time_reversal_list)}", verbosity, 2)
            selected_symmetries = np.where(np.logical_not(time_reversal_list))[0]
            log_message(f"Selected symmetries: {selected_symmetries}", verbosity, 2)
            # Remove time-reversal symmetries
            rotations = [rotations[i] for i in selected_symmetries]
            translations = [translations[i] for i in selected_symmetries]
            time_reversal_list = [time_reversal_list[i] for i in selected_symmetries]



        #         symmetries.append(SymmetryOperation(
        #             rotations[isym],
        #             translations[isym],
        #             real_lattice,
        #             ind=isym + 1,
        #             spinor=spinor,
        #             translation_mod1=translation_mod_1,
        #             time_reversal=time_reversal_list[isym]))

        sg = CLS(
            Lattice=real_lattice,
            spinor=spinor,
            rotations=rotations,
            translations=translations,
            time_reversals=time_reversal_list,
            translation_mod1=translation_mod1,
            number_str=number_str,
            name=name,
            spinor_rotations=None,  # No SOC in this class
            refUC=refUC,
            shiftUC=shiftUC,
            magnetic=magnetic,
            positions=positions,
            typat=typat,
            alat=alat,
            verbosity=verbosity,
        )
        if CLS.__name__ == "SpaceGroupIrreps":
            sg.set_irreptables(verbosity=verbosity,
                               **kwargs_tables,)
        return sg

    @property
    def number(self):
        '''
        To get the number of the space group as an int. Used in WannierBerri. 
        Returns -1 for magnetic spacegroups
        '''
        numbers = self.number_str.split('.')
        if len(numbers) > 1:  # magnetic SG
            return -1
        else:
            return int(numbers[0])

    @property
    def u_symmetries(self):
        '''
        List of unitary symmetries
        '''
        return [x for x in self.symmetries if not x.time_reversal]

    @property
    def au_symmetries(self):
        '''
        List of antiunitary symmetries
        '''
        if self.magnetic:
            return [x for x in self.symmetries if x.time_reversal]
        else:
            return []



    def write_sym_file(self, filename, alat=None):
        """
        Write symmetry operations to a file.

        Parameters
        ----------
        filename : str
            Name of the file.
        alat : float, default=None
            Lattice parameter in angstroms. If not specified, the lattice 
            parameter is not written to the file.
        """

        if alat is None:
            if hasattr(self, 'alat'):
                alat = self.alat
        if alat is None:
            warnings.warn("Lattice parameter not specified. Symmetry operations will be written assuming A=1")
            alat = 1
        with open(filename, "w") as f:
            f.write(f" {len(self.symmetries)} \n")
            for symop in self.symmetries:
                f.write(symop.str_sym(alat))

    @classmethod
    def parse_files(
        CLS,
        fWAV=None,
        fWFK=None,
        prefix=None,
        calculator_gpaw=None,
        fPOS=None,
        spinor=None,
        code="vasp",
        verbosity=0,
        alat=None,
        from_sym_file=None,
        magmom=None,
        include_TR=False,
        ############
        symprec=1e-5,
        angle_tolerance=-1,
        mag_symprec=-1,
        ############
        **kwargs_tables
    ):
        """
        Parse files to create a space-group object.
        Parameters
        ----------
        fWAV : str, default=None
            Name of the file with wavefunctions (e.g. WAVECAR for VASP).
        fWFK : str, default=None
            Name of the file with wavefunctions (e.g. WFK for ABINIT).
        prefix : str, default=None
            Prefix of the files (e.g. prefix for Quantum Espresso).
        calculator_gpaw : GPAW calculator, default=None     
            GPAW calculator object. If provided, it is used to parse the 
            wavefunctions.
        fPOS : str, default=None
            Name of the file with positions (e.g. POSCAR for VASP).
        spinor : bool, default=None
            `True` if wave-functions are spinors (SOC), `False` if they are scalars.
            If not specified, it is determined from the code.
        code : str, default="vasp"
            Code used to generate the files. Supported codes: "vasp", "abinit", 
            "espresso", "wannier90", "gpaw". If not specified, it is assumed to be "vasp".
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing.
        alat : float, default=None
            Lattice parameter in angstroms (quantum espresso convention).
            If not specified, it is determined from the code.
        from_sym_file : str, default=None
            If provided, the symmetry operations are read from this file.
            (format of pw2wannier90 prefix.sym  file)
        magmom : array, default=None
            Each element is the magnetic moment of an ion. If None, non-magnetic calculation.
            If True, magnetic moments are set to zero, i.e. time-reversal symmetry is included in the spacegroup.
        include_TR : bool, default=False
            If `True`, the time-reversal symmetries are included in the space-group.
            If `False`, the time-reversal symmetries are removed from the space-group.
        symprec, angle_tolerance, mag_symprec: float
            see `get_symmetry` and 'get_magnetic_symmetry` in 
            `Spglib documentation <https://spglib.readthedocs.io/en/stable/api/python-api.html#spglib.spglib.get_magnetic_symmetry>'__
        **kwargs_tables : dict
            Additional keyword arguments to pass to the `SpaceGroupIrrep.set_irreptables` method.

        Returns
        -------
        SpaceGroup
            An instance of the `SpaceGroup` class with the parsed symmetry operations.
        """
        code = code.lower()

        if code == "vasp":
            if spinor is None:
                log_message("Spinor is not specified (for VASP), assuming non-spinor calculation", verbosity, 2)
                spinor = False
            parser = ParserVasp(fPOS, fWAV, onlysym=True, verbosity=verbosity)
            Lattice, positions, typat = parser.parse_poscar()

        elif code == "abinit":
            parser = ParserAbinit(fWFK)
            (nband, NK, Lattice, Ecut0, spinor, typat, positions, EF_in) = \
                parser.parse_header(verbosity=verbosity)


        elif code == "espresso":
            parser = ParserEspresso(prefix)
            spinor = parser.spinor
            # alat is saved to be used to write the prefix.sym file
            Lattice, positions, typat, _alat = parser.parse_lattice()
            if alat is None:
                alat = _alat
            spinpol, Ecut0, EF_in, NK, NBin_list = parser.parse_header()


        elif code == "wannier90":
            parser = ParserW90(prefix, unk_formatted=None)
            NK, NBin, spinor, EF_in = parser.parse_header()
            Lattice, positions, typat, kpred = parser.parse_lattice()

        elif code == "gpaw":
            parser = ParserGPAW(calculator=calculator_gpaw,
                                spinor=False if spinor is None else spinor)
            NBin, kpred, Lattice, spinor, typat, positions, EF_in = parser.parse_header()

        else:
            raise RuntimeError(f"Unknown/unsupported code :{code}")


        return CLS.from_cell(
            real_lattice=Lattice,
            positions=positions,
            typat=typat,
            spinor=spinor,
            alat=alat,
            from_sym_file=from_sym_file,
            magmom=magmom,
            include_TR=include_TR,
            verbosity=verbosity,
            ############
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            mag_symprec=mag_symprec,
            ####################
            **kwargs_tables
        )







def read_sym_file(fname):
    """
    Read symmetry operations from a file.

    Parameters
    ----------
    fname : str
        Name of the file.

    Returns
    -------
    np.array(Nsym, 3, 3)
        Each element is a 3x3 array describing a rotation matrix in cartesian coordinates
    np.array(Nsym, 3)
        Each element is a 3D vector describing a translation in units of alat.
    """

    with open(fname, "r") as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    nsym = int(lines[0][0])
    assert len(lines) == 1 + 4 * nsym
    RT = np.array(lines[1:], dtype=float).reshape(nsym, 4, 3)
    rotations = RT[:, 0:3]  # .swapaxes(1, 2)
    translations = RT[:, 3]
    return rotations, translations


def cart_to_crystal(rot_cart, trans_cart, lattice, alat):
    """
    Convert rotation and translation matrices from cartesian to crystal 
    coordinates.

    Parameters
    ----------
    rot_cart : array, shape=(Nsym, 3, 3)
        Each element is a 3x3 array describing a rotation matrix in cartesian
        coordinates
    trans_cart : array, shape=(Nsym, 3)
        Each element is a 3D vector describing a translation in cartesian
        coordinates
    lattice : array, shape=(3, 3)
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    alat : float, default=1
        Lattice parameter in angstroms (quantum espresso convention).

    Returns
    -------
    array, shape=(Nsym, 3, 3)
        Each element is a 3x3 array describing a rotation matrix in crystal 
        coordinates (should be integers)
    array, shape=(Nsym, 3)
        Each element is a 3D vector describing a translation in crystal 
        coordinates
    """

    lat_inv = np.linalg.inv(lattice)
    rot_crystal = np.array([(lat_inv.T @ rot @  lattice.T) for rot in rot_cart])
    assert np.allclose(rot_crystal, np.round(rot_crystal)), f"rotations are not integers in crystal coordinates : {rot_crystal}"
    rot_crystal = np.round(rot_crystal).astype(int)
    trans_crystal = - trans_cart @ lat_inv * alat * BOHR
    return rot_crystal, trans_crystal


# For compatibility with old code
SpaceGroupBare = SpaceGroup
