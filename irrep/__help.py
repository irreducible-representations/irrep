#!/usr/bin/env python3


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

def __help():
    from .__version import __version__ as version
    print ( """ 

           # ###   ###   #####  ### 
           # #  #  #  #  #      #  #
           # ###   ###   ###    ### 
           # #  #  #  #  #      #   
           # #   # #   # #####  #   
        
     version {}  

      calculates the expectation values of symmetry operations 
    <Psi_nk | T(g) | Psi_nk >  as well as irreducible representations,
     Wannier charge centers (1D) Zak phases and many more

   this help is under construction and is far from being complete. 
   If you have interest in the code and not sure how to use it, do not hesitate to contact the author
     
   usage: 
       python3 -m irrep  option=value .. 

    -h   print this help message

    code  vasp (default) or abinit or espresso
    
    ZAK - calcualte Zak phase
    WCC - calcualte  Wannier Charge Centers
    onlysym  - only calculate the symmetry operations
    
    plotbands  -  write gnuplottable files with all symmetry eigenvalues
    
    fWAV  input wavefunctions file  name for vasp. default: WAVECAR
    fPOS  input POSCAR file  name. default: WAVECAR
    fWFK  input wavefunctions file  name for abinit. default: none
    prefix - for QuantumEspresso calculations (data should be in prefix.save)
    
    spinor   wether the wavefunctions are spinor. (relevant and mandatory for VASP only)

    refUC - the lattice vectors of the "reference" unit cell ( as given in the crystallographic tables)
            expressed in therm of the unit cell vectors used in the calculation .
              nine comma-separated numbers
    shiftUC - the vector to shift the calculational unit cell origin (in units of the calculational lattice), to get the unit cell as in the table.
              3 coma-separated numbers

    NB   number of bands in the output. If NB<=0 all bands are used. default:  0
    kpoints   coma-separated list of k-point indices (starting from 1)
    
    kpnames    coma-separated list of k-point names (as in the tables) one entry per each value in kpoints list. (!!!important!!! kpoints is assumed as an ordered list!!!)
    
    IBstart  starting band in the output. If <=0 starting from the lowest band (count from one)  default: 0
    IBend    last     band in the output. If <=0 up to  the highest band (count from one)  default: 0

    Ecut    Energy cut-off, used in the calculation. 

    isymsep   index of the symmetry to separate the eigenstates  ( works well only for norm-conserving potentials (in abinit) )
    
    EXAMPLES:
       python3 -m irrep Ecut=50 code=abinit fWFK=Bi_WFK refUC=0,-1,1,1,0,-1,-1,-1,-1  kpoints=11 IBend=5  kpnames="GM" 
       python3 -m irrep Ecut=50 code=espresso prefix=Bi refUC=0,-1,1,1,0,-1,-1,-1,-1  kpoints=11 IBend=5  kpnames="GM" 
    
    NOTE FOR ABINIT:  always use "istwfk=1"
""".format(version))
    exit()


