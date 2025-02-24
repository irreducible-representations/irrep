import pyfplo.fedit as fedit
import pyfplo.fploio as fploio
import sys

file = sys.argv[1]
fio = fploio.FPLOInput(file)
par = fio.parser()
d = par()
print(f'space group in =.in: {int(d("spacegroup.number").S)}')
