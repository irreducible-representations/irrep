import numpy as np
from sys import stdout

from ..gvectors import Hartree_eV
from ..utility import FortranFileR as FFR
from ..utility import BOHR, log_message
from .common import ParserCommon


class ParserAbinit(ParserCommon):
    """Parser for Abinit WFK files."""

    def __init__(self, filename):
        super().__init__()
        self.fWFK = FFR(filename)
        self.kpt_count = 0

    def parse_header(self, verbosity=0):
        try:
            record = self.fWFK.read_record("S6,2i4", rec=0)
        except Exception as err:
            print(f"Error reading header of Abinit WFK file: {err}")
            record = self.fWFK.read_record( "S8,2i4", rec=0)
        stdout.flush()

        codsvn = record[0][0].decode("ascii").strip()
        headform, fform = record[0][1]
        defversion = ["8.6.3", "9.6.2", "8.4.4", "8.10.3"]
        if codsvn not in defversion:
            log_message(
                f"WARNING, the version {codsvn} of abinit is not in {defversion} and may not be fully tested",
                verbosity,
                1,
            )
        if headform < 80:
            raise ValueError(f"Head form {headform}<80 is not supported")

        record = self.fWFK.read_record("18i4,19f8,4i4", rec=1)[0]
        stdout.flush()
        (bandtot, natom, nkpt, nsym, npsp, nsppol, ntypat, usepaw, nspinor, occopt) = np.array(record[0])[
            [0, 4, 8, 12, 13, 11, 14, 17, 10, 15]
        ]
        rprimd = record[1][7:16].reshape((3, 3)) * BOHR
        ecut = record[1][0] * Hartree_eV
        nshiftk_orig = record[2][1]
        nshiftk = record[2][2]

        if nsppol != 1:
            raise RuntimeError(f"Only nsppol=1 is supported. found {nsppol}")
        if occopt == 9:
            raise RuntimeError("occopt=9 is not supported.")
        if nspinor == 2:
            spinor = True
        elif nspinor == 1:
            spinor = False
        else:
            raise RuntimeError(f"Unexpected value nspinor = {nspinor}")

        fmt = (
            f"{nkpt}i4,{nkpt * nsppol}i4,{nkpt}i4,{npsp}i4,{nsym}i4,"
            f"({nsym},3,3)i4,{natom}i4,({nkpt},3)f8,{bandtot}f8,"
            f"({nsym},3)f8,{ntypat}f8,{nkpt}f8"
        )
        record = self.fWFK.read_record(fmt, rec=2)[0]
        typat = record[6]
        kpt = record[7]
        nband = record[1]
        istwfk = set(record[0])
        npwarr = record[2]

        if istwfk != {1}:
            raise ValueError(f"istwfk should be 1 for all kpoints. Found {istwfk}")
        assert np.sum(nband) == bandtot, "Probably a bug in Abinit"

        record = self.fWFK.read_record(f"f8,({natom},3)f8,f8,f8,{ntypat}f8", rec=3)[0]
        xred = record[1]
        efermi = record[3] * Hartree_eV

        fmt = f"i4,i4,f8,f8,i4,(3,3)i4,(3,3)i4,({nshiftk_orig},3)f8,({nshiftk},3)f8"
        record = self.fWFK.read_record(fmt, rec=4)[0]

        for ipsp in range(npsp):
            record = self.fWFK.read_record("S132,f8,f8,5i4,S32", rec=5 + ipsp)[0]

        self.num_rec_header = 5 + npsp
        if usepaw == 1:
            self.num_rec_header += 2

        self.nband = nband
        self.spinor = spinor
        self.npwarr = npwarr
        self.kpt = kpt
        return (nband, nkpt, rprimd, ecut, spinor, typat, xred, efermi)

    def get_kpt_coord(self, ik):
        return self.kpt[ik]

    def get_record_start(self, ik):
        nband = sum(self.nband[:ik])
        return self.num_rec_header + nband + ik * 3

    def parse_kpoint(self, ik, getWF=True, getE=True):
        nspinor = 2 if self.spinor else 1
        irec_start = self.get_record_start(ik)
        record = self.fWFK.read_record("i4", rec=irec_start)
        npw, nspinor_loc, nband = record
        assert npw == self.npwarr[ik], ("Different number of plane waves in header and k-point's block. Probably a bug in Abinit...")
        assert nspinor_loc == nspinor, ("Different values of nspinor in header and k-point's block. Probably a bug in Abinit...")
        assert nband == self.nband[ik], ("Different number of bands in header and k-point's block. Probably a bug in Abinit...")

        if getE:
            record = self.fWFK.read_record("f8", rec=irec_start + 2)
            eigen = record[:nband]
            eigen *= Hartree_eV
        else:
            eigen = None

        if getWF:
            kg = self.fWFK.read_record("i4", rec=irec_start + 1).reshape(npw, 3)
            WF = np.zeros((nband, npw, nspinor), dtype=complex)
            for iband in range(nband):
                record = self.fWFK.read_record("f8", rec=irec_start + 3 + iband)
                WF[iband, :] = (
                    record[0::2] + 1.0j * record[1::2]
                ).reshape((npw, nspinor), order="F")
        else:
            WF = None
            kg = None

        return WF, eigen, kg
