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
            record = self.fWFK.read_record("S6,2i4")
        except Exception as err:
            print(f"Error reading header of Abinit WFK file: {err}")
            self.fWFK.goto_record(0)
            record = self.fWFK.read_record( "S8,2i4")
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

        record = self.fWFK.read_record("18i4,19f8,4i4")[0]
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
        record = self.fWFK.read_record(fmt)[0]
        typat = record[6]
        kpt = record[7]
        nband = record[1]
        istwfk = set(record[0])
        npwarr = record[2]

        if istwfk != {1}:
            raise ValueError(f"istwfk should be 1 for all kpoints. Found {istwfk}")
        assert np.sum(nband) == bandtot, "Probably a bug in Abinit"

        record = self.fWFK.read_record(f"f8,({natom},3)f8,f8,f8,{ntypat}f8")[0]
        xred = record[1]
        efermi = record[3] * Hartree_eV

        fmt = f"i4,i4,f8,f8,i4,(3,3)i4,(3,3)i4,({nshiftk_orig},3)f8,({nshiftk},3)f8"
        record = self.fWFK.read_record(fmt)[0]

        for ipsp in range(npsp):
            record = self.fWFK.read_record("S132,f8,f8,5i4,S32")[0]

        if usepaw == 1:
            self.fWFK.read_record("i4")
            self.fWFK.read_record("i4")

        self.nband = nband
        self.spinor = spinor
        self.npwarr = npwarr
        self.kpt = kpt
        return (nband, nkpt, rprimd, ecut, spinor, typat, xred, efermi)

    def parse_kpoint(self, ik):
        nspinor = 2 if self.spinor else 1

        for _ in range(self.kpt_count, ik + 1):
            if self.kpt_count < ik:
                skip = True
            else:
                skip = False

            record = self.fWFK.read_record("i4")
            npw, nspinor_loc, nband = record
            assert npw == self.npwarr[ik], ("Different number of plane waves in header and k-point's block. Probably a bug in Abinit...")
            assert nspinor_loc == nspinor, ("Different values of nspinor in header and k-point's block. Probably a bug in Abinit...")
            assert nband == self.nband[ik], ("Different number of bands in header and k-point's block. Probably a bug in Abinit...")

            kg = self.fWFK.read_record("i4").reshape(npw, 3)

            record = self.fWFK.read_record("f8")
            if not skip:
                eigen = record[:nband]
                eigen *= Hartree_eV

            if skip:
                record = self.fWFK.read_record("f8")
            else:
                WF = np.zeros((nband, npw, nspinor), dtype=complex)
                for iband in range(nband):
                    record = self.fWFK.read_record("f8")
                    WF[iband, :] = (
                        record[0::2] + 1.0j * record[1::2]
                    ).reshape((npw, nspinor), order="F")

            self.kpt_count += 1

        return WF, eigen, kg
