#!/usr/bin/env python

import numpy as np
from irrep.bandstructure import BandStructure
from gpaw import GPAW, PW
from ase import Atoms
from ase.parallel import paropen, world


def calc_Bi():
    c = 7.46567804
    a= 4.9620221
    ca=c/a
    r3 = np.sqrt(3)
    lattice = np.array([[r3/2, 1/2, ca],
                        [-r3/2, 1/2, ca],
                        [0, -1, ca]])*a*0.529
    #     [[ 4.29723723  2.48101107  7.46567804]
    #    [-4.29723723  2.48101107  7.46567804]
    #    [ 0.         -4.96202214  7.46567804]] 
    x = 0.237
    Bi2 = Atoms(symbols='Bi2',
                scaled_positions=[(-x, -x, -x),
                                  ( x, x, x)],
                cell=lattice,
                pbc=True)
    calc = GPAW(
                nbands=100,
                mode=PW(300),
                kpts = {'size': (2,2,2),'gamma':True},
                # symmetry='off',
                txt='Bi.txt')
    
    Bi2.calc = calc
    Bi2.get_potential_energy()
    calc.write('Bi.gpw', mode='all')
    return calc

try:
    calc = GPAW("Bi.gpw")
except:
    calc = calc_Bi()

# from gpaw.spinorbit import soc_eigenstates

if world.rank == 0:

    print ("Fermi level",calc.get_fermi_level())
    bandstructure = BandStructure(code="gpaw", calculator_gpaw=calc,
                                  spinor=True,
                                  Ecut=30,degen_thresh=1e-4,
                                  calculate_traces=True)
    print ("Bandstructure",bandstructure)
    bandstructure.write_trace()
    print("done")


    print (calc.get_bz_k_points().shape)
    print (calc.get_ibz_k_points().shape)


