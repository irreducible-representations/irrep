#!/usr/bin/env python

from gpaw import GPAW
import irrep
from irrep.bandstructure import BandStructure
# from ase import Atoms
# from ase.dft.wannier import Wannier


if False:
    a = 4.4570000
    c = 5.9581176
    x = 0.274

    te = Atoms(symbols='Te3',
               scaled_positions =[( x, 0, 0),
                                  ( 0, x, 1./3),
                                  (-x,-x, 2./3)],
               cell=(a, a, c, 90, 90, 120),
               pbc=True)


    calc = GPAW(nbands=16,
                mode="pw",
            kpts = {'size': (3, 3, 4),'gamma':True},
            symmetry='off',
            txt='Te.txt')

    te.calc = calc
    te.get_potential_energy()
    calc.write('Te_pw.gpw', mode='all')

calc = GPAW("Te_pw.gpw")

bandstructure = BandStructure(code="gpaw", calculator_gpaw=calc,Ecut=30)
exit()

print(calc.get_ibz_k_points())

wan = Wannier(nwannier=3, calc=calc, fixedstates=3)

wan.localize() # Optimize rotation to give maximal localization
wan.save('wannier-s.pickle') # Save localization and rotation matrix


# Re-load using saved wannier data
#wan = Wannier(nwannier=18, calc=calc, fixedstates=15, file='file.pickle')


