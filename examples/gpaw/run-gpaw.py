#!/usr/bin/env python


#parallel run with :  gpaw -P 16 python run-gpaw.py 

import numpy as np
from irrep.bandstructure import BandStructure
from gpaw import GPAW, PW
from ase import Atoms
from ase.parallel import world


c = 7.46567804
a = 4.9620221
ca = c / a
r3 = np.sqrt(3)
lattice = np.array([[r3 / 2, 1 / 2, ca],
                    [-r3 / 2, 1 / 2, ca],
                    [0, -1, ca]]) * a * 0.529
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
    mode=PW(300),
    kpts={'size': (6, 6, 6), 'gamma': True},
    # symmetry='off',
    txt='Bi.txt')

# calc = GPAW("Bi.gpw")

Bi2.calc = calc
Bi2.get_potential_energy()

calc_bands = calc.fixed_density(
    nbands=20,
    kpts={'size': (1, 1, 1), 'gamma': True},
    txt='Bi_gamma.txt')



calc.write('Bi.gpw', mode='all')

calc_bands.write('Bi-gamma.gpw', mode='all')
