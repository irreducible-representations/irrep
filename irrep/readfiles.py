
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


import numpy as np
import scipy
from scipy.io import FortranFile as FF
from sys import stdout
from .utility import FortranFileR as FFR


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

    def __init__(self, filename, RL=3):
        self.f = open(filename, "rb")
        self.rl = 3
        # RECLENGTH=3 # the length of a record in WAVECAR. It is defined in the
        # first record, so let it be 3 fo far"
        self.rl, ispin, iprec = [int(x) for x in self.record(0)]
        if iprec != 45200:
            raise RuntimeError("double precision WAVECAR is not supported")
        if ispin != 1:
            raise RuntimeError(
                "WAVECAR contains spin-polarized non-spinor wavefunctions. "
                + "ISPIN={0}  this is not supported yet".format(ispin)
            )

    def record(self, irec, cnt=np.Inf, dtype=float):
        """An auxilary function to get records from WAVECAR"""
        self.f.seek(irec * self.rl)
        return np.fromfile(self.f, dtype=dtype, count=min(self.rl, cnt))


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


Rydberg_eV = 13.605693  # eV
Hartree_eV = 2 * Rydberg_eV


class ParserAbinit():
    """
    Parse header of the WFK file of Abinit.

    Paramters
    ---------
    filename : str
        Name of the WFK file of Abinit.

    Attributes
    ----------
    fWFK : file object
        Corresponds to the WFK file.
    rprim : array, shape=(3,3)
        Each row contains cartesian coordinates of a basis vector forming the 
        unit-cell in real space.
    ecut : float
        Plane-wave cutoff (in eV) to consider in the expansion of 
        wave-functions.
    usepaw : int
        1 if pseudopotentials are PAW, 0 otherwise.
    typat : array
        Each element is a number identifying the atomic species of an ion. 
        atomic species of an ion. See `cell` parameter of function 
        `get_symmetry` in 
        `Spglib <https://spglib.github.io/spglib/python-spglib.html#get-symmetry>`_.
    kpt : array
        Each row contains the direct coordinates of a k-point.
    nband : array
        Each element contains the number of bands in a k-point.
    xred : array
        Each row contains the direct coordinates of an ion's position. 
    efermi : float
        Fermi-level.

    Notes
    -----
    Abinit's routines that write the header can be found
    `here <https://docs.abinit.org/guide/abinit/#5>`_ or in the file
    `src/56_io_mpi/m_hdr.f90`.
    """

    def __init__(self, filename):
        #fWFK = FF(fname, "r")
        self.fWFK = FFR(filename)  # temporary
        (self.nband,
         self.nkpt,
         self.rprimd,
         self.ecut,
         self.spinor,  # keep as attribute
         self.typat,
         self.xred,
         self.kpt,
         self.efermi,
         self.npwarr,
         self.usepaw) = self.parse_header(filename)
        self.kpt_count = 0  # index of the next k-point to be read

    def parse_header(self, filename):

        # 1st record
            # write(unit=header) codvsn,headform,fform
        try:  # version < 9.0.0
            record = record_abinit(self.fWFK, 'a6,2i4')
        except:  # version > 9.0.0
            self.fWFK.goto_record(0)
            record = record_abinit(self.fWFK, 'a8,2i4')
        stdout.flush()

        # Check version number of Abinit
        codsvn = record[0][0].decode('ascii').strip()
        headform, fform = record[0][1]
        defversion = ['8.6.3', '9.6.2', '8.4.4', '8.10.3']
        if codsvn not in defversion:
            print(("WARNING, the version {0} of abinit is not in {1} "
                   "and may not be fully tested"
                   .format(codsvn, defversion))
                  )
        if headform < 80:
            raise ValueError(
                "Head form {0}<80 is not supported".format(headform)
                )

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
        rprimd = record[1][7:16].reshape((3, 3))
        ecut = record[1][0] * Hartree_eV
        nshiftk_orig = record[2][1]
        nshiftk = record[2][2]

        # Check spin-polarization and if wave functions are spinors
        if nsppol != 1:
            raise RuntimeError(
                "Only nsppol=1 is supported. found {0}".format(nsppol)
                )
        if occopt == 9:  # extra records since Abinit v9
            raise RuntimeError("occopt=9 is not supported.")
        if nspinor == 2:
            spinor = True
        elif nspinor == 1:
            spinor = False
        else:
            raise RuntimeError(
                "Unexpected value nspinor = {0}".format(nspinor)
                )

        # 3rd record
            # write(unit=header) istwfk(1:nkpt),nband(1:nkpt*nsppol),&
            # & npwarr(1:nkpt),so_psp(1:npsp),symafm(1:nsym), &
            # & symrel(1:3,1:3,1:nsym),typat(1:natom), kpt(1:3,1:nkpt), &
            # & occ(1:bantot),tnons(1:3,1:nsym),znucltypat(1:ntypat), &
            # & wtk(1:nkpt)
        fmt = ('{nkpt}i4,{n2}i4,{nkpt}i4,{npsp}i4,{nsym}i4,({nsym},3,3)i4,'
               '{natom}i4,({nkpt},3)f8,{bandtot}f8,({nsym},3)f8,{ntypat}f8,'
               '{nkpt}f8'
               .format(
                    nkpt=nkpt,
                    n2=nkpt * nsppol,
                    npsp=npsp,
                    nsym=nsym,
                    natom=natom,
                    bandtot=bandtot,
                    ntypat=ntypat)
               )
        record = record_abinit(self.fWFK, fmt)[0]
        typat = record[6]
        kpt = record[7]
        nband = record[1]
        istwfk = record[0]
        npwarr = record[2]

        # istwfk and npwarr are int, should be set, array and array
        if nkpt == 1:
            istwfk = set([istwfk])
            npwarr = np.array([npwarr])
            nband  = np.array([nband])
        else:
            istwfk = set(self.istwfk)

        # Check that istwfk was 1 and consistency of number of bands
        if istwfk != {1}:
            raise ValueError(("istwfk should be 1 for all kpoints. Found {0}"
                              .format(istwfk))
                             )
        assert np.sum(nband) == bandtot, "Probably a bug in Abinit"

        # 4th record
            # write(unit,err=10, iomsg=errmsg) hdr%residm, hdr%xred(:,:), &
            # & hdr%etot, hdr%fermie, hdr%amu(:)
        record = record_abinit(
                    self.fWFK,
                    'f8,({natom},3)f8,f8,f8,{ntypat}f8'.format(natom=natom,
                                                               ntypat=ntypat
                                                               )
                    )[0]
        xred = record[1]
        efermi = record[3] * Hartree_eV

        # 5th record: skip it
            # write(unit,err=10, iomsg=errmsg) &
            # & hdr%kptopt, hdr%pawcpxocc, hdr%nelect, hdr%charge, &
            # & hdr%icoulomb, hdr%kptrlatt,hdr%kptrlatt_orig, &
            # & hdr%shiftk_orig(:,1:hdr%nshiftk_orig), &
            # & hdr%shiftk(:,1:hdr%nshiftk)
        fmt = ('i4,i4,f8,f8,i4,(3,3)i4,(3,3)i4,({nshiftkorig},3)f8,'
               '({nshiftk},3)f8'
               .format(nshiftkorig=nshiftk_orig, nshiftk=nshiftk)
               )
        record = record_abinit(self.fWFK,fmt)[0]

        # 6th record: skip it
            # read(unit, err=10, iomsg=errmsg) hdr%title(ipsp), &
            # & hdr%znuclpsp(ipsp), hdr%zionpsp(ipsp), hdr%pspso(ipsp), &
            # & hdr%pspdat(ipsp), hdr%pspcod(ipsp), hdr%pspxc(ipsp), &
            # & hdr%lmn_size(ipsp), hdr%md5_pseudos(ipsp)
        for ipsp in range(npsp):
            record = record_abinit(self.fWFK, "a132,f8,f8,5i4,a32")[0]

        # 7th record: additional records if usepaw=1
        if usepaw == 1:
            record_abinit(self.fWFK,"i4")
            record_abinit(self.fWFK,"i4")

        # Set as attributes quantities that need to be retrieved from outside the class
        return (nband, nkpt, rprimd, ecut, spinor, typat, xred, kpt, 
                efermi, npwarr, usepaw)

    def parse_kpoint(self, ik):

        print("Reading k-point", ik)
        nspinor = 2 if self.spinor else 1

        # We need to skip lines in fWFK until we reach the lines of ik
        for i in range(self.kpt_count, ik+1):

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
                eigen, occ = record[:nband], record[nband:]
                eigen *= Hartree_eV

            # 4th record: coefficients of expansions in plane waves
            if skip:
                record = record_abinit(self.fWFK, "f8")
            else:
                CG = np.zeros((nband, npw * nspinor), dtype=complex)
                for iband in range(nband):
                    record = record_abinit(self.fWFK, "f8")
                    CG[iband] = record[0::2] + 1.0j * record[1::2]

            self.kpt_count += 1

        return CG, eigen, kg

        # Check orthonormality for norm-conserving pseudos
        #if self.usepaw == 0:
        #    largest_value = np.max(np.abs(CG.conj().dot(CG.T)
        #                                  - np.eye(IBend - IBstart)))
        #    assert largest_value < 1e-10, "Wave functions are not orthonormal"


class ParserVasp:

    def __init__(self, fPOS, fWAV):
        self.fPOS = fPOS
        self.fWAV = WAVECARFILE(fWAV)

    def parse_poscar(self):
        """
        Parses POSCAR.

        Returns
        ------
        lattice : array
            3x3 array where cartesian coordinates of basis  vectors **a**, **b** 
            and **c** are given in rows. 
        positions : array
            Each row contains the direct coordinates of an ion's position. 
        numbers : list
            Each element is a number identifying the atomic species of an ion.
        """

        fpos = (l.strip() for l in open(self.fPOS))
        title = next(fpos)
        lattice = float(
            next(fpos)) * np.array([next(fpos).split() for i in range(3)], dtype=float)
        try:
            nat = np.array(next(fpos).split(), dtype=int)
        except BaseException:
            nat = np.array(next(fpos).split(), dtype=int)

        numbers = [i + 1 for i in range(len(nat)) for j in range(nat[i])]

        l = next(fpos)
        if l[0] in ['s', 'S']:
            l = next(fpos)
        cartesian=False
        if l[0].lower()=='c':
            cartesian=True
        elif l[0].lower()!='d':
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
                print(err)
                pass
        if sum(nat) != i:
            raise RuntimeError(
                "not all atomic positions were read : {0} of {1}".format(
                    i, sum(nat)))
        if cartesian:
            positions = positions.dot(np.linalg.inv(lattice))
        return lattice, positions, numbers

    def parse_header(self):
        tmp = self.fWAV.record(1)
        NK = int(tmp[0])
        NBin = int(tmp[1])
        Ecut0 = tmp[2]
        lattice = np.array(tmp[3:12]).reshape(3, 3)
        return NK, NBin, Ecut0, lattice
