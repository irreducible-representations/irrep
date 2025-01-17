import pyfplo.fedit as fedit
import pyfplo.fploio as fploio
import sys

num_sg = int(sys.argv[1])
if len(sys.argv) > 2:
    tol_wp = float(sys.argv[2])
else:
    tol_wp = 1e-4

def printsettings():
    fio=fploio.FPLOInput('=.in')
    par=fio.parser()
    d=par()
    print( 'spacegroup number : ',d('spacegroup.number').S)
    print( 'spacegroup setting: ',d('spacegroup.setting').S)
    print( 'lattice constants : ',d('lattice_constants').listS)
    print( 'axis angle        : ',d('axis_angles').listS)
    dw=d('wyckoff_positions')
    print( 'Wyckoff positions: ',dw.size())
    for i in range(dw.size()):
        taus=dw[i]('tau').listS
        print( '{0:>2s} {1:>20s} {2:>20s} {3:>20s}'.format(dw[i]('element').S,taus[0],taus[1],taus[2]))


if __name__ == '__main__':

    # Create =.in if it doesn't exits, parse cif and write into =.in
    fio=fploio.FPLOInput('=.in')
    fio.structureFromCIFFile(f'{num_sg}.cif', wyckofftolerance=tol_wp, determinesymmetry=True)
    fio.writeFile("=.in")
    
    # Change other parameters in =.in
    fed=fedit.Fedit(recreate=False)
    fed.iteration(n=100)
    fed.relativistic('s')  # no SOC
    fed.vxc('5')
    fed.pipeFedit()
    
    # Print info about crystal
    printsettings()
