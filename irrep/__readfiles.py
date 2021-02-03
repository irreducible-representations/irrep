
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


class WAVECARFILE:

    def __init__(self, fname=None, RL=3):
        self.f = open(fname, "rb")
        self.rl = 3

    def record(self, irec, cnt=np.Inf, dtype=float):
        "an auxilary function to get records from WAVECAR"
        self.f.seek(irec * self.rl)
        return np.fromfile(self.f, dtype=dtype, count=min(self.rl, cnt))


def record_abinit(fWFK, st):
    "an auxilary function to get records from WAVECAR"
    r = fWFK.read_record(st)
    if scipy.__version__.split(".")[0] == "0":
        r = r[0]
    return r


Rydberg_eV = 13.605693  # eV
Hartree_eV = 2 * Rydberg_eV


class AbinitHeader():

    def __init__(self, fname):

        self.fWFK = FF(fname, "r")
        fWFK = self.fWFK
        record = fWFK.read_record('a6,2i4')
#    print (record)
        stdout.flush()
        codsvn = record[0][0].decode('ascii')
        headform, fform = record[0][1]
        defversion = '8.6.3 '
        if not (codsvn == defversion):
            print(
                "WARNING, the version {0} of abinit is not {1}".format(
                    codsvn, defversion))
        if headform < 80:
            raise ValueError(
                "Head form {0}<80 is not supported".format(headform))

        record = fWFK.read_record('18i4,19f8,4i4')[0]
        # write(unit=header) codvsn,headform,fform
        # write(unit=header) bantot,date,intxc,ixc,natom,ngfft(1:3),&
        # & nkpt,nspden,nspinor,nsppol,nsym,npsp,ntypat,occopt,pertcase,usepaw,&
        # & ecut,ecutdg,ecutsm,ecut_eff,qptn(1:3),rprimd(1:3,1:3),stmbias,tphysel,tsmear,usewvl,
        #hdr%nshiftk_orig, hdr%nshiftk, hdr%mband
#    print (record)
        self.rprimd = record[1][7:16].reshape((3, 3))
#    print (self.rprimd)
        self.ecut = record[1][0] * Hartree_eV
        print("ecut={0} eV".format(self.ecut))
        stdout.flush()
        bandtot, self.natom, self.nkpt, nsym, npsp, nsppol, ntypat, self.usepaw, nspinor = np.array(
            record[0])[[0, 4, 8, 12, 13, 11, 14, 17, 10]]
        if nsppol != 1:
            raise RuntimeError(
                "Only nsppol=1 is supported. found {0}".format(nsppol))
        if nspinor == 2:
            self.spinor = True
        elif nspinor == 1:
            self.spinor = False
        else:
            raise RuntimeError(
                "Unexpected value nspinor = {0}".format(nspinor))
        #if usepaw==1: raise ValueError("usepaw==1 not implemented")
        nshiftk_orig = record[2][1]
        nshiftk = record[2][2]
        #print (bandtot,natom,nkpt,nsym,npsp,nsppol,ntypat)
        record = fWFK.read_record(
            '{nkpt}i4,{n2}i4,{nkpt}i4,{npsp}i4,{nsym}i4,({nsym},3,3)i4,{natom}i4,({nkpt},3)f8,{bandtot}f8,({nsym},3)f8,{ntypat}f8,{nkpt}f8'.format(
                nkpt=self.nkpt,
                n2=self.nkpt *
                nsppol,
                npsp=npsp,
                nsym=nsym,
                natom=self.natom,
                bandtot=bandtot,
                ntypat=ntypat))[0]
        #print (record)
        # write(unit=header) istwfk(1:nkpt),nband(1:nkpt*nsppol),&
        # & npwarr(1:nkpt),so_psp(1:npsp),symafm(1:nsym),symrel(1:3,1:3,1:nsym),typat(1:natom),&
        # & kpt(1:3,1:nkpt),occ(1:bantot),tnons(1:3,1:nsym),znucltypat(1:ntypat),wtk(1:nkpt)
#    wtk=record[11]
        self.npwarr = record[2]
        istwfk = record[0]
        if set(istwfk) != {1}:
            raise ValueError(
                "istwfk should be 1 for all kpoints. found {0}".format(istwfk))
        self.typat = record[6]
        self.kpt = record[7]

        self.nband = record[1]
        assert(np.sum(self.nband) == bandtot)
        #print (kpt,nband)

        record = fWFK.read_record(
            'f8,({natom},3)f8,f8,f8,{ntypat}f8'.format(
                natom=self.natom, ntypat=ntypat))[0]
        self.xred = record[1]
        self.efermi = record[3]
        # write(unit,err=10, iomsg=errmsg) hdr%residm, hdr%xred(:,:), hdr%etot, hdr%fermie, hdr%amu(:)
        #print (record)

        record = fWFK.read_record(
            "i4,i4,f8,f8,i4,(3,3)i4,(3,3)i4,({nshiftkorig},3)f8,({nshiftk},3)f8".format(
                nshiftkorig=nshiftk_orig, nshiftk=nshiftk))[0]
        #record=fWFK.read_record("i4,i4,f8,f8,i4,i4,(3,3)f8,5f8,i4".format(nshiftkorig=nshiftk_orig,nshiftk=nshiftk) )[0]
        ##record=fWFK.read_record("i4,i4,f8,f8,i4,9i4,9i4,6f8".format(nshiftkorig=nshiftk_orig,nshiftk=nshiftk) )[0]
        # write(unit,err=10, iomsg=errmsg) &
        #   hdr%kptopt, hdr%pawcpxocc, hdr%nelect, hdr%charge, hdr%icoulomb,&
        #   hdr%kptrlatt,hdr%kptrlatt_orig, hdr%shiftk_orig(:,1:hdr%nshiftk_orig),hdr%shiftk(:,1:hdr%nshiftk)
        #print (record)

        for ipsp in range(npsp):
            record = fWFK.read_record("a132,f8,f8,5i4,a32")[0]
        #   read(unit, err=10, iomsg=errmsg) &
        # &   hdr%title(ipsp), hdr%znuclpsp(ipsp), hdr%zionpsp(ipsp), hdr%pspso(ipsp), hdr%pspdat(ipsp), &
        # &   hdr%pspcod(ipsp), hdr%pspxc(ipsp), hdr%lmn_size(ipsp), hdr%md5_pseudos(ipsp)
        #    print (record)

        if self.usepaw == 1:
            record = fWFK.read_record('i4')
#        print(record)
            record = fWFK.read_record('f8')
#        print(record)
