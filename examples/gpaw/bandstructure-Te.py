# web-page: bandstructure.png
"""Band structure tutorial

Calculate the band structure of Si along high symmetry directions
Brillouin zone
"""
# P1
from ase.build import bulk
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac

# Perform standard ground state calculation (with plane wave basis)
# si = bulk('Si', 'diamond', 5.43)
from ase.dft.kpoints import bandpath

a = 4.4570000
c = 5.9581176
x = 0.274
te = Atoms(symbols='Te3',
            scaled_positions =[( x, 0, 0),
                                ( 0, x, 1./3),
                                (-x,-x, 2./3)],
            cell=(a, a, c, 90, 90, 120),
            pbc=True)


def groundstate():
    calc = GPAW(mode=PW(100),
                xc='HSE06',
                kpts=(3, 3, 4),
                random=True,  # random guess (needed if many empty bands required)
                occupations=FermiDirac(0.01),
                txt='Te_gs.txt')

    te.calc = calc
    te.get_potential_energy()
    ef = calc.get_fermi_level()
    calc.write('Te_gs_hse.gpw')

groundstate()
# P2
# Restart from ground state and fix potential:

points = [[1/3, 1/3, 0], [1/3,1/3,1/2]]
# points = [[0,0, 0], [0,0,1]]


kpts, x, X = bandpath(points, te.cell, npoints=51)

calc = GPAW('Te_gs_hse.gpw').fixed_density(
    nbands=16,
    symmetry='off',
    kpts=kpts,
    convergence={'bands': 8})

calc.write('Te_bands_hse.gpw', mode='all')    

# P3
bs = calc.band_structure()
bs.plot(filename='bandstructure.png', show=True, emax=10.0)
