
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


import os
import numpy as np
import scipy
import h5py
from scipy.io import FortranFile as FF
from sys import stdout

from .gvectors import calc_gvectors, Hartree_eV
from .utility import FortranFileR as FFR
from .utility import str2bool, BOHR, split, log_message
import xml.etree.ElementTree as ET


class WAVECARFILE:
    """
    Routines to read info from file WAVECAR of VASP.

    Parameters
    ----------
    fname : str, default=None
        Name of the WAVECAR file.
    RL : int, default=3
        Length parameter used to locate info in the file.

    Attributes
    ----------
    f : file object
        Corresponds to `fname`.
    rl : int
        Equal to parameter `RL`.
    """

    def __init__(self, filename, RL=3, verbosity=0):
        self.verbosity = verbosity
        self.f = open(filename, "rb")
        self.rl = RL
        # RECLENGTH=3 # the length of a record in WAVECAR. It is defined in the
        # first record, so let it be 3 fo far"
        self.rl, ispin, iprec = [int(x) for x in self.record(0)]
        self.iprec = iprec
        log_message(f"iprec tag = {iprec}, record_length = {self.rl} bytes", self.verbosity, 1)
        if iprec not in (45200, 53300):
            raise RuntimeError(f"invalid iprec tag found: {iprec}, probably not a single-precision file. Double-precision is not supported")
        if ispin != 1:
            raise RuntimeError("WAVECAR contains spin-polarized non-spinor wavefunctions."
                               f"ISPIN={ispin}  this is not supported yet")
        self.nrec_enocc = None  # will be set later
        self.nrec_kpoint = None  # will be set later
        self.nrec_header = 2

    def set_nrec_kpoint(self, NBin):
        size_enocc = (4 + 3 * NBin) * 8
        # number of records needed to store band energies and occupations
        self.nrec_enocc = (size_enocc + self.rl - 1) // self.rl
        log_message(f"number of records for energies and occupancies : {self.nrec_enocc}", self.verbosity, 1)
        if self.iprec in (42200, 42210):
            assert self.nrec_enocc == 1, (f"energies and occupancies for tag {self.iprec} should fit in one record. However, "
                                        f"the record length is {self.rl} bytes, which does not fit 4 + 3*{NBin} = {(4 + 3 * NBin)}*8 = {size_enocc} bytes for {NBin} bands")
        self.nrec_kpoint = NBin + self.nrec_enocc

    def record(self, irec, cnt=np.inf, dtype=float):
        """An auxilary function to get records from WAVECAR"""
        self.f.seek(irec * self.rl)
        return np.fromfile(self.f, dtype=dtype, count=min(self.rl, cnt))

    def irec_start_k(self, ik):
        return self.nrec_header + ik * self.nrec_kpoint

    def record_k_header(self, ik):
        rec_start = self.irec_start_k(ik)
        return np.hstack([self.record(rec_start + i) for i in range(self.nrec_enocc)])

    def record_k_band(self, ik, ib, cnt=np.inf):
        irec = self.irec_start_k(ik) + self.nrec_enocc + ib
        return self.record(irec, cnt=cnt, dtype=np.complex64)



def record_abinit(fWFK, st):
    """
    An auxilary function to get records from WAVECAR

    Parameters
    ----------
    fWFK : file object
        Corresponds to the WFK file of Abinit.
    st : str
        Format to be read.

    Returns
    -------
    r : str
        String read.
    """
    r = fWFK.read_record(st)
    if scipy.__version__.split(".")[0] == "0":
        r = r[0]
    return r



class ParserAbinit():
    """
    Parse header of the WFK file of Abinit.

    Parameters
    ---------
    filename : str
        Name of the WFK file of Abinit.

    Attributes
    ----------
    fWFK : file object
        Corresponds to the WFK file.
    kpt : array
        Each row contains the direct coordinates of a k-point.
    nband : array
        Each element contains the number of bands in a k-point.
    spinor : bool
        Whether the DFT calculation involved spinors (SOC) or not
    npwarr : array
        Each element is the number of plane waves used at a k-point
    kpt : array
        Each row contains the coordinates of a k-point in the DFT BZ

    Notes
    -----
    Abinit's routines that write the header can be found
    `here <https://docs.abinit.org/guide/abinit/#5>`_ or in the file
    `src/56_io_mpi/m_hdr.f90`.
    """

    def __init__(self, filename):
        # fWFK = FF(fname, "r")
        self.fWFK = FFR(filename)  # temporary
        self.kpt_count = 0  # index of the next k-point to be read

    def parse_header(self, verbosity=0):
        '''
        Parse header of WFK file and save as attributes quantities that 
        will be used in the rest of methods

        Parameters
        ----------
        verbosity : int, default=0
            Verbosity level. Default set to minimal printing

        Returns
        -------
        nband : array
            Each element contains the number of bands in a k-point.
        nkpt : int
            Number of k-points in WFK file
        rprimd : array
            Each row contains the Cartesian coords of a DFT cell vector
        ecut : float
            Plane-wave cutoff used for the DFT calculation
        spinor : bool
            Whether the DFT calculation involved spinors (SOC) or not
        typat : array
            Each element is an integer expressing the type ion
        xred : array
            Each row contains the direct coords of an ion
        efermi : float
            Fermi energy
        '''

        # 1st record
        # write(unit=header) codvsn,headform,fform
        try:  # version < 9.0.0
            record = record_abinit(self.fWFK, 'a6,2i4')
        except Exception:  # version > 9.0.0
            self.fWFK.goto_record(0)
            record = record_abinit(self.fWFK, 'a8,2i4')
        stdout.flush()

        # Check version number of Abinit
        codsvn = record[0][0].decode('ascii').strip()
        headform, fform = record[0][1]
        defversion = ['8.6.3', '9.6.2', '8.4.4', '8.10.3']
        if codsvn not in defversion:
            log_message(f"WARNING, the version {codsvn} of abinit is not in {defversion} "
                        "and may not be fully tested", verbosity, 1)
        if headform < 80:
            raise ValueError(f"Head form {headform}<80 is not supported")

        # 2nd record
            # write(unit=header) bantot,date,intxc,ixc,natom,ngfft(1:3), nkpt,&
            # & nspden,nspinor,nsppol,nsym,npsp,ntypat,occopt,pertcase,usepaw,&
            # & ecut,ecutdg,ecutsm,ecut_eff,qptn(1:3),rprimd(1:3,1:3),stmbias,&
            # & tphysel,tsmear,usewvl, hdr%nshiftk_orig, hdr%nshiftk, hdr%mband
        record = record_abinit(self.fWFK, '18i4,19f8,4i4')[0]
        stdout.flush()
        (bandtot,
         natom,
         nkpt,
         nsym,
         npsp,
         nsppol,
         ntypat,
         usepaw,
         nspinor,
         occopt) = np.array(record[0])[[0, 4, 8, 12, 13, 11, 14, 17, 10, 15]]
        rprimd = record[1][7:16].reshape((3, 3)) * BOHR
        ecut = record[1][0] * Hartree_eV
        nshiftk_orig = record[2][1]
        nshiftk = record[2][2]

        # Check spin-polarization and if wave functions are spinors
        if nsppol != 1:
            raise RuntimeError(f"Only nsppol=1 is supported. found {nsppol}")
        if occopt == 9:  # extra records since Abinit v9
            raise RuntimeError("occopt=9 is not supported.")
        if nspinor == 2:
            spinor = True
        elif nspinor == 1:
            spinor = False
        else:
            raise RuntimeError(f"Unexpected value nspinor = {nspinor}")

        # 3rd record
            # write(unit=header) istwfk(1:nkpt),nband(1:nkpt*nsppol),&
            # & npwarr(1:nkpt),so_psp(1:npsp),symafm(1:nsym), &
            # & symrel(1:3,1:3,1:nsym),typat(1:natom), kpt(1:3,1:nkpt), &
            # & occ(1:bantot),tnons(1:3,1:nsym),znucltypat(1:ntypat), &
            # & wtk(1:nkpt)
        fmt = (
            f"{nkpt}i4,{nkpt * nsppol}i4,{nkpt}i4,{npsp}i4,{nsym}i4,"
            f"({nsym},3,3)i4,{natom}i4,({nkpt},3)f8,{bandtot}f8,"
            f"({nsym},3)f8,{ntypat}f8,{nkpt}f8"
        )
        record = record_abinit(self.fWFK, fmt)[0]
        typat = record[6]
        kpt = record[7]
        nband = record[1]
        istwfk = record[0]
        npwarr = record[2]

        # istwfk and npwarr are int, should be set, array and array
        if nkpt == 1:
            istwfk = set([int(istwfk)])
            npwarr = np.array([npwarr])
            nband = np.array([nband])
        else:
            istwfk = set(istwfk)

        # Check that istwfk was 1 and consistency of number of bands
        if istwfk != {1}:
            raise ValueError(f"istwfk should be 1 for all kpoints. Found {istwfk}")
        assert np.sum(nband) == bandtot, "Probably a bug in Abinit"

        # 4th record
        # write(unit,err=10, iomsg=errmsg) hdr%residm, hdr%xred(:,:), &
        # & hdr%etot, hdr%fermie, hdr%amu(:)
        record = record_abinit(self.fWFK, f"f8,({natom},3)f8,f8,f8,{ntypat}f8")[0]
        xred = record[1]
        efermi = record[3] * Hartree_eV

        # 5th record: skip it
        # write(unit,err=10, iomsg=errmsg) &
        # & hdr%kptopt, hdr%pawcpxocc, hdr%nelect, hdr%charge, &
        # & hdr%icoulomb, hdr%kptrlatt,hdr%kptrlatt_orig, &
        # & hdr%shiftk_orig(:,1:hdr%nshiftk_orig), &
        # & hdr%shiftk(:,1:hdr%nshiftk)
        fmt = f"i4,i4,f8,f8,i4,(3,3)i4,(3,3)i4,({nshiftk_orig},3)f8,({nshiftk},3)f8"
        record = record_abinit(self.fWFK, fmt)[0]

        # 6th record: skip it
        # read(unit, err=10, iomsg=errmsg) hdr%title(ipsp), &
        # & hdr%znuclpsp(ipsp), hdr%zionpsp(ipsp), hdr%pspso(ipsp), &
        # & hdr%pspdat(ipsp), hdr%pspcod(ipsp), hdr%pspxc(ipsp), &
        # & hdr%lmn_size(ipsp), hdr%md5_pseudos(ipsp)
        for ipsp in range(npsp):
            record = record_abinit(self.fWFK, "a132,f8,f8,5i4,a32")[0]

        # 7th record: additional records if usepaw=1
        if usepaw == 1:
            record_abinit(self.fWFK, "i4")
            record_abinit(self.fWFK, "i4")

        # Set as attributes quantities that need to be retrieved by the rest of methods
        self.nband = nband
        self.spinor = spinor
        self.npwarr = npwarr
        self.kpt = kpt
        return (nband, nkpt, rprimd, ecut, spinor, typat, xred, efermi)

    def parse_kpoint(self, ik):
        '''
        Parse block of a k-point from WFK file

        Returns
        -------
        WF : array
            Each row contains the coefficients of the plane-wave 
            expansion of a wave function
        eigen : array
            Energies of the wave functions
        kg : array
            Each row contains the direct coords of a reciprocal vector 
            used in the expansion of wave functions
        '''

        nspinor = 2 if self.spinor else 1

        # We need to skip lines in fWFK until we reach the lines of ik
        for i in range(self.kpt_count, ik + 1):

            if self.kpt_count < ik:
                skip = True
            else:
                skip = False

            # 1st record: npw, nspinor, nband
            record = record_abinit(self.fWFK, "i4")  # [0]
            npw, nspinor_loc, nband = record
            assert npw == self.npwarr[ik], ("Different number of plane waves "
                                            "in header and k-point's block. "
                                            "Probably a bug in Abinit...")
            assert nspinor_loc == nspinor, ("Different values of nspinor in "
                                            "header and k-point's block. "
                                            "Probably a bug in Abinit...")
            assert nband == self.nband[ik], ("Different number of bands in "
                                             "header and k-point's block. "
                                             "Probably a bug in Abinit...")

            # 2nd record: reciprocal lattice vectors in the expansion
            kg = record_abinit(self.fWFK, "i4").reshape(npw, 3)

            # 3rd record: energies and occupations
            record = record_abinit(self.fWFK, "f8")
            if not skip:
                eigen = record[:nband]
                eigen *= Hartree_eV

            # 4th record: coefficients of expansions in plane waves
            if skip:
                record = record_abinit(self.fWFK, "f8")
            else:
                WF = np.zeros((nband, npw, nspinor), dtype=complex)
                for iband in range(nband):
                    record = record_abinit(self.fWFK, "f8")
                    WF[iband, :] = (record[0::2] + 1.0j * record[1::2]).reshape((npw, nspinor), order='F')

            self.kpt_count += 1

        return WF, eigen, kg

        # Check orthonormality for norm-conserving pseudos
        # if self.usepaw == 0:
        #    largest_value = np.max(np.abs(CG.conj().dot(CG.T)
        #                                  - np.eye(IBend - IBstart)))
        #    assert largest_value < 1e-10, "Wave functions are not orthonormal"


class ParserVasp:
    """
    Parser for Vasp interface

    Parameters
    ---------
    fPOS : str
        Name of the POSCAR file.
    fWAV : str
        Name of the WAVECAR file.
    onlysym : bool
        To stop right after parsing the POSCAR, before parsing the 
        header of the WAVECAR.

    Attributes
    ----------
    fPOS : str
        Name of the POSCAR file.
    fWAV : class
        Instance of `WAVECARFILE`
    """

    def __init__(self, fPOS, fWAV, onlysym=False, verbosity=0):
        self.verbosity = verbosity
        self.fPOS = fPOS
        if not onlysym:
            self.fWAV = WAVECARFILE(fWAV, verbosity=self.verbosity)


    def parse_poscar(self):
        """
        Parses POSCAR.

        Parameters
        ----------
        verbosity : int, default=0
            Verbosity level. Default set to minimal printing

        Returns
        ------
        lattice : array
            3x3 array where cartesian coordinates of basis  vectors **a**, **b** 
            and **c** are given in rows. 
        positions : array
            Each row contains the direct coordinates of an ion's position. 
        typat : list
            Each element is a number identifying the atomic species of an ion.
        """

        log_message(f'Reading POSCAR: {self.fPOS}', self.verbosity, 1)
        fpos = (l.strip() for l in open(self.fPOS))
        title = next(fpos)  # title
        del title
        lattice = float(
            next(fpos)) * np.array([next(fpos).split() for i in range(3)], dtype=float)
        try:
            nat = np.array(next(fpos).split(), dtype=int)
        except BaseException:
            nat = np.array(next(fpos).split(), dtype=int)

        typat = [i + 1 for i in range(len(nat)) for j in range(nat[i])]

        l = next(fpos)
        if l[0] in ['s', 'S']:
            l = next(fpos)
        cartesian = False
        if l[0].lower() == 'c':
            cartesian = True
        elif l[0].lower() != 'd':
            raise RuntimeError(
                'only "direct" or "cartesian"atomic coordinates are supproted')
        positions = np.zeros((np.sum(nat), 3))
        i = 0
        for l in fpos:
            if i >= sum(nat):
                break
            try:
                positions[i] = np.array(l.split()[:3])
                i += 1
            except Exception as err:
                log_message(err, self.verbosity, 1)
                pass
        if sum(nat) != i:
            raise RuntimeError(f"not all atomic positions were read : {i} of {sum(nat)}")
        if cartesian:
            positions = positions.dot(np.linalg.inv(lattice))
        return lattice, positions, typat

    def parse_header(self):
        '''
        Parse header of WAVECAR

        Returns
        -------
        NK : int
            Number of k-points
        NBin : int
            Number of bands
        Ecut0 : float
            Plane-wave cutoff (in eV) used in DFT
        lattice : array
            Each row contains the cartesian coords of a unit cell vector
        '''

        tmp = self.fWAV.record(1)
        NK = int(tmp[0])
        NBin = int(tmp[1])
        self.fWAV.set_nrec_kpoint(NBin=NBin)
        Ecut0 = tmp[2]
        lattice = np.array(tmp[3:12]).reshape(3, 3)
        return NK, NBin, Ecut0, lattice

    def parse_kpoint(self, ik, NBin, spinor):
        '''
        Parse block of a particular k-point from WAVECAR

        Parameters
        ----------
        ik : int
            Index of the k-point
        NBin : int
            Number of bands
        spinor : bool
            Whether wave functions are spinors (SOC)

        Returns
        -------
        Energy : array
            Energy levels. Degenerate levels are repeated
        kpt : array
            Direct coords of the k-points with respect to the basis vectors 
            of the DFT reciprocal space cell
        npw : int
            Number of plane waves in the expansion of wave functions
        '''

        r = self.fWAV.record_k_header(ik)
        nspinor = 2 if spinor else 1
        # Check if number of plane waves is even for spinors
        npw = int(r[0])
        log_message(f"npw = {npw}, nspinor = {nspinor}, NBin = {NBin}", self.verbosity, 2)
        if spinor:
            assert npw % 2 == 0, f"odd number of coefs {npw} for spinor wavefunctions"
        npw //= nspinor
        kpt = r[1:4]
        Energy = np.array(r[4: 4 + NBin * 3]).reshape(NBin, 3)[:, 0]
        WF = np.zeros((NBin, npw, nspinor), dtype=np.complex64)
        for ib in range(NBin):
            WF[ib] = self.fWAV.record_k_band(ik=ik, ib=ib, cnt=npw * nspinor).reshape((npw, nspinor), order='F')
        return WF, Energy, kpt, npw


class ParserEspresso:
    '''
    Parser of the interface for Quantum Espresso

    Parameters
    ----------
    prefix : str
        Prefix that serves as path to the `.save` directory. For example: 
        if the path is `./foo/bar.save` then `prefix` is `foo/bar`.

    Attributes
    ----------
    prefix : str
        Prefix that serves as path to the `.save` directory.
    input : class
        Instance of `Element` in the ElementTree XML API corresponding 
        to tag `input` in `data-file-schema.xml` file
    bandstructure : class
        Instance of `Element` in the ElementTree XML API corresponding 
        to tag `band_structure` in `data-file-schema.xml` file
    spinor : bool
        Whether wave functions are spinors (SOC)
    '''

    def __init__(self, prefix):
        self.prefix = prefix
        mytree = ET.parse(prefix + ".save/data-file-schema.xml")
        myroot = mytree.getroot()

        self.input = myroot.find("input")
        outp = myroot.find("output")
        self.bandstr = outp.find("band_structure")

        # todo: define spinor as property with getter
        self.spinor = str2bool(self.bandstr.find("noncolin").text)


    def parse_header(self):
        '''
        Parse universal info of the bandstructure from `data-file-schema.xml` 
        file

        Returns
        -------
        spinpol : bool
            Whether the calculation is spin-polarized
        Ecut0 : float
            Plane-wave cutoff (in eV) used in DFT
        EF : float
            Fermi energy in eV
        NK : int
            Number of k-points
        NBin_list : list
            If the calculation was spin polarized, `NBin_list[0]` and 
            `NBin_list[1]` are the number of bands for spin up and down 
            channels, respectively. Otherwise, `NBin_list[0]` is the 
            number of bands.
        '''

        Ecut0 = float(self.input.find("basis").find("ecutwfc").text)
        Ecut0 *= Hartree_eV
        NK = len(self.bandstr.findall("ks_energies"))

        # Parse number of bands
        try:
            NBin_dw = int(self.bandstr.find('nbnd_dw').text)
            NBin_up = int(self.bandstr.find('nbnd_up').text)
            spinpol = True
            print(f"spin-polarised bandstructure composed of {NBin_up} up and {NBin_dw} dw states")
            NBin_dw + NBin_up
        except AttributeError:
            spinpol = False
            NBin = int(self.bandstr.find('nbnd').text)

        try:
            EF = float(self.bandstr.find("fermi_energy").text) * Hartree_eV
        except Exception:
            EF = None

        if spinpol:
            NBin_list = [NBin_up, NBin_dw]
        else:
            NBin_list = [NBin]
        return spinpol, Ecut0, EF, NK, NBin_list

    def parse_lattice(self):
        '''
        Parse info about the crystal structure from `data-file-schema.xml` 
        file

        Returns
        -------
        lattice : array
            Each row contains the cartesian coords of a DFT unit cell vector
        positions : array
            Each row contains the direct coords of an ion in the DFT cell
        typat : list
            Indices that describe the type of element at each position. 
            All ions of the same type share the same index.
        alat : float
            Lattice parameter in Quantum Espresso's convention
        '''

        ntyp = int(self.input.find("atomic_species").attrib["ntyp"])
        struct = self.input.find("atomic_structure")
        nat = int(struct.attrib["nat"])
        alat = float(struct.attrib["alat"])
        del nat

        # Parse lattice vectors
        lattice = []
        for i in range(3):
            lattice.append(struct.find("cell").find(f"a{i + 1}").text.strip().split())
        lattice = BOHR * np.array(lattice, dtype=float)

        # Parse atomic positions in cartesian coordinates
        positions = []
        for at in struct.find("atomic_positions").findall("atom"):
            positions.append(at.text.split())
        positions = np.dot(np.array(positions, dtype=float) * BOHR,
                      np.linalg.inv(lattice))

        # Parse indices denoting type of atom
        atnames = []
        for sp in self.input.find("atomic_species").findall("species"):
            atnames.append(sp.attrib["name"])
        if len(atnames) != ntyp:
            raise RuntimeError("Error in the assigment of atom species. "
                               "Probably a bug in Quantum Espresso, but "
                               "your DFT input files")
        atnumbers = {atn: i + 1 for i, atn in enumerate(atnames)}
        typat = []
        for at in struct.find("atomic_positions").findall("atom"):
            typat.append(atnumbers[at.attrib["name"]])

        return lattice, positions, typat, alat


    def parse_kpoint(self, ik, NBin, spin_channel, verbosity=0):
        '''
        Parse block of a particular k-point from `data-file-schema.xml` file

        Parameters
        ----------
        ik : int
            Index of the k-point
        NBin : int
            Number of bands
        spin_channel : str
            `up` for spin up, `dw` for spin down, `None` if not spin polarized
        verbosity : int, default=0
            Verbosity level. Default set to minimalistic printing


        Returns
        -------
        WF : array
            Each row contains the coefficients of the plane-wave expansion of 
            a wave function
        Energy : array
            Energy levels in eV. Degenerate levels are repeated
        kg : array
            Each row contains the direct coords (ints) of a plane wave's 
            reciprocal lattice vector
        kpt : array
            Direct coords of the k-point w.r.t. DFT cell vectors
        '''

        kptxml = self.bandstr.findall("ks_energies")[ik]

        # Parse energy levels
        Energy = np.array(kptxml.find("eigenvalues").text.split(), dtype=float)
        Energy *= Hartree_eV
        npw = int(kptxml.find("npw").text)
        nspinor = 2 if self.spinor else 1
        npwtot = npw * nspinor

        # Open file with the wave functions
        wfcname = f"wfc{'' if spin_channel is None else spin_channel}{ik + 1}"
        fWFC = None
        checked_files = []
        for extension in ["hdf5", "dat"]:
            for strcase in [str.lower, str.upper]:
                filename = f"{self.prefix}.save/{strcase(wfcname)}.{extension}"
                if os.path.exists(filename):
                    if extension == "hdf5":
                        fWFC = h5py.File(filename, 'r')
                        attrnames = ['gamma_only', 'igwx', 'ik', 'ispin', 'nbnd', 'ngw', 'npol', 'scale_factor', 'xk']
                        attributes = []
                        for atr in attrnames:
                            attributes.append(fWFC.attrs[atr])

                        _gamma_only, _igwx, ik, _ispin, _nbnd, _ngw, _npol, _scale_factor, xk = attributes
                        kpt = np.array(xk)
                        Miller_Indices = fWFC['MillerIndices']
                        B = np.array([Miller_Indices.attrs[f'bg{i}'] for i in range(1, 4)])
                        kg = np.array(Miller_Indices[::])
                        kpt = kpt.dot(np.linalg.inv(B))
                        # Parse coefficients of wave functions
                        evc = np.array(fWFC['evc'], dtype=float)
                        WF = evc[:, 0::2] + 1.0j * evc[:, 1::2]
                    else:
                        fWFC = FF(filename, "r")
                        rec = record_abinit(fWFC, "i4,3f8,i4,i4,f8")[0]
                        kpt = rec[1]  # cartesian coords of k-point

                        rec = record_abinit(fWFC, "4i4")
                        igwx = rec[1]

                        # Determine direct coords of k-point
                        rec = record_abinit(fWFC, "(3,3)f8")
                        B = np.array(rec)
                        rec = record_abinit(fWFC, f"({igwx},3)i4")
                        kg = np.array(rec)
                        log_message(f'npwtot: {npwtot}, igwx: {igwx}', verbosity, 2)
                        kpt = kpt.dot(np.linalg.inv(B))
                        # Parse coefficients of wave functions
                        WF = np.zeros((NBin, npwtot), dtype=complex)
                        for ib in range(NBin):
                            rec = record_abinit(fWFC, f"{npwtot * 2}f8")
                            WF[ib] = rec[0::2] + 1.0j * rec[1::2]
                    WF = WF.reshape((NBin, npw, nspinor), order='F')
                    return WF, Energy, kg, kpt
                checked_files.append(filename)
        raise RuntimeError(f"Wavefunction file not found. Tried files: {checked_files}")


class ParserW90:
    '''
    Parser for Wannier90's interface

    Parameters
    ----------
    prefix : str
        Part of the path that serves as prefix for the `wannier90.win` file.
        For example: if the path is `./foo/bar.win`, then `prefix` is 
        `foo/bar`

    Attributes
    ----------
    prefix : str
        Part of the path that serves as prefix for the `wannier90.win` file.
    fwin : list
        Each element is a list of a non-comment line in `wannier90.win` file,
        split by blank spaces
    ind : list
        Each element is the first keyword of a line in `wannier90.win`
    iterwin : iterator object
        Iterator object for attribute `fwin`
    NBin : int
        Number of DFT bands
    spinor : bool
        Whether wave functions are spinors (SOC)
    NK : int
        Number of k-points in DFT calculation
    '''

    def __init__(self, prefix, unk_formatted=False, spin_channel=None):

        self.prefix = prefix
        self.spin_channel = spin_channel
        self.path = os.path.dirname(prefix)
        self.fwin = [l.strip().lower() for l in open(prefix + ".win").readlines()]
        self.fwin = [
            [s.strip() for s in split(l)]
            for l in self.fwin
            if len(l) > 0 and l[0] not in ("!", "#")
        ]
        self.ind = np.array([l[0] for l in self.fwin])
        self.iterwin = iter(self.fwin)
        self.unk_formatted = unk_formatted

    # TODO : use wannier90io instead
    def parse_header(self):
        '''
        Parse universal properties of the band structure

        Returns
        -------
        NK : int
            Number of k-points in DFT calculation
        NBin : int
            Number of DFT bands
        spinor : bool
            Whether wave functions are spinors (SOC)
        EF : float
            Fermi energy in eV
        '''

        self.NBin = self.get_param("num_bands", int)
        self.spinor = str2bool(self.get_param("spinors", str))
        try:
            EF = self.get_param("fermi_energy", float, None)
        except RuntimeError:
            EF = None
        self.NK = np.prod(np.array(self.get_param("mp_grid", str).split(), dtype=int))

        return self.NK, self.NBin, self.spinor, EF


    def parse_lattice(self):
        '''
        Parse crystal structure from `wannier90.win` file

        Returns
        -------
        lattice : array
            Each row contains the cartesian coords of a DFT unit cell vector
        positions : array
            Each row contains the direct coords of an ion in the DFT cell
        typat : list
            Indices that describe the type of element at each position. 
            All ions of the same type share the same index.
        kpred : array
            Each row contains the direct coords of a k-point in the DFT cell
        '''

        # Initialize quantities to make sure they aren't twice in .win
        lattice = None
        kpred = None
        found_atoms = False

        for l in self.iterwin:

            if l[0].startswith("begin"):

                # Parse lattice vectors
                if l[1] == "unit_cell_cart":
                    if lattice is not None:
                        raise RuntimeError(f"'begin unit_cell_cart' found more than once in {self.prefix}.win")
                    l1 = next(self.iterwin)
                    if l1[0] in ("bohr", "ang"):
                        units = l1[0]
                        L = [next(self.iterwin) for i in range(3)]
                    else:
                        units = "ang"
                        L = [l1] + [next(self.iterwin) for i in range(2)]
                    lattice = np.array(L, dtype=float)
                    if units == "bohr":
                        lattice *= BOHR
                    self.check_end("unit_cell_cart")

                # Parse k-points
                elif l[1] == "kpoints":
                    if kpred is not None:
                        raise RuntimeError(f"'begin kpoints' found more then once in {self.prefix}.win")
                    kpred = np.zeros((self.NK, 3), dtype=float)
                    for i in range(self.NK):
                        kpred[i] = next(self.iterwin)[:3]
                    self.check_end("kpoints")

                # Parse atomic positions
                elif l[1].startswith("atoms_"):
                    if l[1][6:10] not in ("cart", "frac"):
                        raise RuntimeError(f"unrecognised block :  '{l[0]}' ")
                    if found_atoms:
                        raise RuntimeError(f"'begin atoms_***' found more then once  in {self.prefix}.win")
                    found_atoms = True
                    positions = []
                    nameat = []
                    while True:
                        l1 = next(self.iterwin)
                        if l1[0] == "end":
                            if l1[1] != l[1]:
                                raise RuntimeError(f"'{' '.join(l)}' ended with 'end {l1[1]}'")
                            else:
                                break
                        nameat.append(l1[0])
                        positions.append(l1[1:4])
                    typatdic = {n: i + 1 for i, n in enumerate(set(nameat))}
                    typat = [typatdic[n] for n in nameat]
                    positions = np.array(positions, dtype=float)
                    if l[1][6:10] == "cart":  # from cartesian to direct coords
                        positions = positions.dot(np.linalg.inv(lattice))

        return lattice, positions, typat, kpred


    def parse_energies(self):
        '''
        Parse energies from `wannier90.eig` file

        Returns
        -------
        Energy : array
            Each row contains the energy levels of a k-point. Degenerate 
            levels are repeated
        '''

        feig = self.prefix + ".eig"
        Energy = np.loadtxt(self.prefix + ".eig")
        try:
            if Energy.shape[0] != self.NBin * self.NK:
                raise RuntimeError("wrong number of entries ")
            ik = np.array(Energy[:, 1]).reshape(self.NK, self.NBin)
            if not np.all(
                ik == np.arange(1, self.NK + 1)[:, None] * np.ones(self.NBin, dtype=int)[None, :]
            ):
                raise RuntimeError("wrong k-point indices")
            ib = np.array(Energy[:, 0]).reshape(self.NK, self.NBin)
            if not np.all(
                ib == np.arange(1, self.NBin + 1)[None, :] * np.ones(self.NK, dtype=int)[:, None]
            ):
                raise RuntimeError("wrong band indices")
            Energy = Energy[:, 2].reshape(self.NK, self.NBin)
        except Exception as err:
            raise RuntimeError(f" error reading {feig} : {err}")
        return Energy

    def parse_kpoint(self, ik, selectG):
        '''
        Parse wave functions' file of a k-point

        Parameters
        ----------
        ik : int
            Index of the k-point
        selectG : array
            First 3 rows of the array returned by :func:`~gvectors.calc_gvectors`

        Returns
        -------
        WF : array
            Each row contains the coefficients of the expansion of a wave 
            function in plane waves
        '''
        if self.unk_formatted:
            return self.parse_kpoint_formatted(ik, selectG)
        else:
            return self.parse_kpoint_unformatted(ik, selectG)

    def get_UNK_name(self, ik):
        if self.spin_channel is None:
            spin_channel_loc = 'NC' if self.spinor else '1'
        else:
            spin_channel_loc = str(self.spin_channel).upper()
        return os.path.join(self.path, f"UNK{ik:05d}.{spin_channel_loc}")

    def parse_kpoint_formatted(self, ik, selectG):
        fname = self.get_UNK_name(ik)
        print(f"parse_kpoint_formatted: {fname}")
        fUNK = open(fname, "r")
        ngx, ngy, ngz, ik_in, nbnd = (int(x) for x in fUNK.readline().split())
        ngtot = ngx * ngy * ngz
        nspinor = 2 if self.spinor else 1
        self.check_ik_nb(ik_in, ik, nbnd, fname)
        # Parse WF coefficients
        WF_in = np.loadtxt(fUNK, dtype=complex)
        WF_in = WF_in[:, 0] + 1.0j * WF_in[:, 1]
        fUNK.close()
        ng_loc = selectG[0].shape[0]
        WF = np.zeros((self.NBin, ng_loc, nspinor), dtype=complex)
        for ib in range(self.NBin):
            for i in range(nspinor):
                i_start = ib * nspinor * ngtot + i * ngtot
                cg_tmp = WF_in[i_start:i_start + ngtot]
                cg_tmp = cg_tmp.reshape((ngx, ngy, ngz), order="F")
                cg_tmp = np.fft.fftn(cg_tmp)
                WF[ib, :, i] = cg_tmp[selectG]
        return WF


    def check_ik_nb(self, ik_in, ik, nbnd, fname):
        """Checks of consistency between UNK and .win"""
        if ik_in != ik:
            raise RuntimeError(f"file {fname} contains point number {ik_in}, expected {ik}")
        if nbnd != self.NBin:
            raise RuntimeError(f"file {fname} contains {nbnd} bands, expected {self.NBin}")


    def parse_kpoint_unformatted(self, ik, selectG):
        fname = self.get_UNK_name(ik)
        fUNK = FF(fname, "r")
        ngx, ngy, ngz, ik_in, nbnd = record_abinit(fUNK, "i4,i4,i4,i4,i4")[0]
        ngtot = ngx * ngy * ngz
        nspinor = 2 if self.spinor else 1

        self.check_ik_nb(ik_in, ik, nbnd, fname)

        # print (f"selectG.shape = {selectG[0].shape}, ngtot = {ngtot}, nspinor = {nspinor}")
        # Parse WF coefficients
        ng_loc = selectG[0].shape[0]
        WF = np.zeros((self.NBin, ng_loc, nspinor), dtype=complex)
        for ib in range(self.NBin):
            for i in range(nspinor):
                cg_tmp = record_abinit(fUNK, f"{ngtot * 2}f8")
                cg_tmp = (cg_tmp[0::2] + 1.0j * cg_tmp[1::2]).reshape(
                    (ngx, ngy, ngz), order="F")
                cg_tmp = np.fft.fftn(cg_tmp)
                WF[ib, :, i] = cg_tmp[selectG]
        return WF

    def parse_grid(self, ik):
        '''
        Parse grid of plane waves for a k-point from the file of 
        wave functions

        Parameters
        ----------
        ik : int
            Index of k-point

        Returns
        -------
        ngx : int
            Number of k-point along 1st direction in reciprocal space 
        ngy : int
            Number of k-point along 2nd direction in reciprocal space 
        ngz : int
            Number of k-point along 3rd direction in reciprocal space 
        '''

        fname = self.get_UNK_name(ik)
        if self.unk_formatted:
            fUNK = open(fname, "r")
            ngx, ngy, ngz, ik_in, nbnd = (int(x) for x in fUNK.readline().split())
            fUNK.close()
        else:
            fUNK = FF(fname, "r")
            ngx, ngy, ngz, ik_in, nbnd = record_abinit(fUNK, "i4,i4,i4,i4,i4")[0]
            fUNK.close()
        return ngx, ngy, ngz

    def check_end(self, name):
        """
        Check if block in .win file is closed.

        Parameters
        ----------
        name : str
            Name of the block in .win file.

        Raises
        ------
        RuntimeError
            Block is not closed.
        """
        s = next(self.iterwin)
        if " ".join(s) != "end " + name:
            raise RuntimeError(f"expected 'end {name}', found {' '.join(s)}")



    def get_param(self, key, tp, default=None, join=False):
        """
        Return value of a parameter in .win file.

        Parameters
        ----------
        key : str
            Wannier90 input parameter.
        tp : function
            Function to apply to the value of the parameter, before 
            returning it.
        default
            Default value to return in case parameter `key` is not found.
        join : bool, default=False
            If the value of parameter `key` contains more than one element, 
            they will be concatenated with a blank space if `join` is set 
            to `True`. Used when the parameter is `mpgrid`.

        Returns
        -------
        Type(`tp`)
            Return the value of the parameter, after applying function 
            passed es keyword `tp`.

        Raises
        ------
        RuntimeError
            The parameter is not found in .win file, it is found more than 
            once or its value is formed by many elements but it is not
            `mpgrid`.
        """
        i = np.where(self.ind == key)[0]
        if len(i) == 0:
            if default is None:
                raise RuntimeError(f"parameter {key} was not found in {self.prefix}.win")
            else:
                return default
        if len(i) > 1:
            raise RuntimeError(f"parameter {key} was found {len(i)} times in {self.prefix}.win")

        x = self.fwin[i[0]][1:]  # mp_grid should work
        if len(x) > 1:
            if join:
                x = " ".join(x)
            else:
                raise RuntimeError(f"length {len(x)} found for parameter {key}, rather than length 1 in {self.prefix}.win")
        else:
            x = self.fwin[i[0]][1]
        return tp(x)


class ParserGPAW:

    """
    Parser for GPAW interface

    Parameters
    ----------
    calculator : str or GPAW
        instance of GPAW class or the name of the file containing it
    """

    def __init__(self, calculator, spinor=False, spin_channel=None):
        from gpaw import GPAW
        if isinstance(calculator, str):
            calculator = GPAW(calculator)
        self.calculator = calculator
        self.nband = self.calculator.get_number_of_bands()
        print("spinor", spinor)
        self.spinor = spinor

        if self.spinor:
            nspins = self.calculator.get_number_of_spins()
            self.spin_channels = np.arange(nspins)
        else:
            if spin_channel is None:
                spin_channel = 0
            self.spin_channels = [spin_channel]


    def parse_header(self):
        kpred = self.calculator.get_ibz_k_points()
        Lattice = self.calculator.atoms.cell
        typat = self.calculator.atoms.get_atomic_numbers()
        positions = self.calculator.atoms.get_scaled_positions()
        EF_in = self.calculator.get_fermi_level()
        return (self.nband * (1 + int(self.spinor)), kpred, Lattice, self.spinor, typat, positions, EF_in)

    def parse_kpoint(self, ik, RecLattice, Ecut):
        WFupdw = [np.array([
            self.calculator.get_pseudo_wave_function(kpt=ik, band=ib, periodic=True, spin=ispin)
            for ib in range(self.nband)]) for ispin in self.spin_channels]
        Eupdw = [self.calculator.get_eigenvalues(kpt=ik, spin=ispin) for ispin in self.spin_channels]

        print(f"shapes of WFupdw: {[wf.shape for wf in WFupdw]}")
        ngx, ngy, ngz = WFupdw[0].shape[1:]
        kpt = self.calculator.get_ibz_k_points()[ik]
        kg, eKG = calc_gvectors(kpt,
                           RecLattice,
                           Ecut,
                           spinor=False,
                           nplanemax=np.max([ngx, ngy, ngz]) // 2
                            )
        selectG = tuple(kg[:, 0:3].T)

        for i in range(len(WFupdw)):
            WFupdw[i] = np.fft.fftn(WFupdw[i], axes=(1, 2, 3))
            WFupdw[i] = np.array([wf[selectG] for wf in WFupdw[i]])
        if self.spinor:
            if len(WFupdw) == 1:
                WFupdw = WFupdw * 2
                Eupdw = Eupdw * 2
            WF = WFupdw[0]
            WFspinor = np.zeros((2 * WF.shape[0], WF.shape[1], 2), dtype=complex)
            H = get_soc_gpaw(self.calculator, ik, flatten=True)
            rng = np.arange(0, 2 * self.nband, 2)
            for s in range(2):
                H[rng + s, rng + s] += Eupdw[s]
            E, V = np.linalg.eigh(H)
            V = V.T
            # e, v_mn = self.soc.eigenvectors()
            for s in range(2):
                WFspinor[:, :, s] = V[:, s::2] @ WFupdw[s]
            energies = E
            WFout = WFspinor
        else:
            WFout = WFupdw[0][:, :, None]
            energies = self.calculator.get_eigenvalues(kpt=ik, spin=self.spin_channels[0])

        return energies, WFout, kg, kpt, eKG



def get_soc_gpaw(calc, ik, flatten=True,
                theta=0, phi=0):
    """
    Calculate the spin-orbit coupling Hamiltonian in the basis of
    Kohn-Sham states, for a given k-point.  
    """
    from ase.units import Hartree
    from gpaw.spinorbit import soc

    nspin = calc.get_number_of_spins()

    C_ss = np.array(
        [
            [
                np.cos(theta / 2) * np.exp(-1.0j * phi / 2),
                -np.sin(theta / 2) * np.exp(-1.0j * phi / 2),
            ],
            [
                np.sin(theta / 2) * np.exp(1.0j * phi / 2),
                np.cos(theta / 2) * np.exp(1.0j * phi / 2),
            ],
        ]
    )

    # sx_ss = np.array([[0, 1], [1, 0]], complex)
    # sy_ss = np.array([[0, -1.0j], [1.0j, 0]], complex)
    # sz_ss = np.array([[1, 0], [0, -1]], complex)

    # s_vss = [
    #     C_ss.T.conj() @ sx_ss @ C_ss,
    #     C_ss.T.conj() @ sy_ss @ C_ss,
    #     C_ss.T.conj() @ sz_ss @ C_ss,
    # ]


    # kd = calc.wfs.kd
    dVL_avii = {
        a: soc(calc.wfs.setups[a], calc.hamiltonian.xc, D_sp)
        for a, D_sp in calc.density.D_asp.items()
    }

    m = calc.get_number_of_bands()
    nk = len(calc.get_ibz_k_points())
    assert ik < nk, f"ik = {ik} >= nk = {nk}"


    h_soc = np.zeros((2, 2, m, m), complex)

    H_a = {}
    for a, dVL_vii in dVL_avii.items():
        ni = dVL_vii.shape[1]
        H_ssii = np.zeros((2, 2, ni, ni), complex)
        H_ssii[0, 0] = dVL_vii[2]
        H_ssii[0, 1] = dVL_vii[0] - 1j * dVL_vii[1]
        H_ssii[1, 0] = dVL_vii[0] + 1j * dVL_vii[1]
        H_ssii[1, 1] = -dVL_vii[2]
        H_a[a] = H_ssii

    q = ik
    for a, H_ssii in H_a.items():
        h_ssii = np.einsum(
            "ab,bcij,cd->adij", C_ss.T.conj(), H_ssii, C_ss, optimize=True
        )
        for s1 in range(2):
            for s2 in range(2):
                h_ii = h_ssii[s1, s2]
                if nspin == 2:
                    P1_mi = calc.wfs.kpt_qs[q][s1].P_ani[a]
                    P2_mi = calc.wfs.kpt_qs[q][s2].P_ani[a]
                if nspin == 1:
                    P1_mi = calc.wfs.kpt_qs[q][0].P_ani[a]
                    P2_mi = calc.wfs.kpt_qs[q][0].P_ani[a]

                h_soc[s1, s2] += np.dot(np.dot(P1_mi.conj(), h_ii), P2_mi.T)
    h_soc[:] *= Hartree
    if flatten:
        h_soc_old = h_soc.copy()
        h_soc = np.zeros((2 * m, 2 * m), complex)
        for s1 in range(2):
            for s2 in range(2):
                h_soc[s1::2, s2::2] = h_soc_old[s1, s2]
    return h_soc
