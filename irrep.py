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

def hlp():
    print ( """ 

           # ###   ###   #####  ### 
           # #  #  #  #  #      #  #
           # ###   ###   ###    ### 
           # #  #  #  #  #      #   
           # #   # #   # #####  #   
        
      calculates the expectation values of symmetry operations 
    <Psi_nk | T(g) | Psi_nk >  as well as irreducible representations,
     Wannier charge centers (1D) Zak phases and many more

   this help is under construction and is far from being complete. 
   If you have interest in the code and not sure how to use it, do not hesitate to contact the author
     
   usage: 
       irrep.py option=value .. 

    -h   print this help message

    code  vasp (default) or abinit
    
    ZAK - calcualte Zak phase
    WCC - calcualte  Wannier Charge Centers
    onlysym  - only calculate the symmetry operations
    
    plotbands  -  write gnuplottable files with all symmetry eigenvalues
    
    fWAV  input wavefunctions file  name for vasp. default: WAVECAR
    fPOS  input POSCAR file  name. default: WAVECAR
    fWFK  input wavefunctions file  name for abinit. default: none
    
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
    
    
    NOTE FOR ABINIT:  always use "istwfk=1"
""")
    exit()



writeTXT=False
import sys
import numpy as np
import datetime
import math
#from gvectors import calc_gvectors,Arot,symm_eigenvalues,NotSymmetryError
from spacegroup import SpaceGroup
import  bandstructure as BS
from aux import str2bool,str2list


print ("The code was called with the following command-line options\n "+"   ".join(sys.argv))

## reading input from command-line

fWAV="WAVECAR"
fWFK=None
fPOS="POSCAR"
IBstart=None
IBend=None
Ecutsym=None
kpoints=None
spinor=None
code="vasp"
refUC=None
shiftUC=None
charfile=None
symmetries=None
kpnames=None
EF=None
isymsep=None
degen_thresh=1e-4
groupKramers=True
ZAK=False
WCC=False
onlysym=False
writebands=True
plotbands=True
plotFile=None
seedname=""

for arg in sys.argv[1:]:
    if arg=="-h": 
        hlp()
    if arg=="ZAK" :
        ZAK=True
    if arg=="WCC" :
        WCC=True
    if arg.lower()=="onlysym" :
        onlysym=True
        spinor=False

    else:
        k=arg.split("=")[0]
        v="=".join(arg.split("=")[1:])
        if   k=="fWAV"  : fWAV=v
        elif k=="fWFK"  : fWFK=v
        elif k=="charfile"  : charfile=v
        elif k=="EF"  : EF=float(v)
#        elif k=="preline"  : preline=v
        elif k=="fPOS"  : fPOS=v
        elif k=="IBend"   : IBend=int(v)
        elif k=="IBstart"   : IBstart=int(v)
        elif k=="isymsep"   : isymsep=str2list(v)
        elif k=="Ecut"   : Ecutsym=float(v)
        elif k=="kpoints"   : kpoints=str2list(v)
        elif k=="degen_thresh"   : degen_thresh=float(v)
        elif k=="code" : code=v.lower()
        elif k=="seedname" : seedname=v
        elif k=="kpnames" :     kpnames=v.split(",")
        elif k=="refUC" : refUC=np.array(v.split(','),dtype=float).reshape( (3,3) ) 
        elif k=="shiftUC" : shiftUC=np.array(v.split(','),dtype=float).reshape( 3 ) 
        elif k=="symmetries" : symmetries=str2list(v)
        elif k=="groupKramers" : groupKramers=str2bool(v)
        elif k=="writebands" : writebands=str2bool(v)
        elif k=="plotbands" : plotbands=str2bool(v)
        elif k=="plotFile" : plotFile=v
        elif k=="spinor"   :   spinor=str2bool(v)


try:
    print ( fWFK.split("/")[0].split("-") )
    preline=" ".join(s.split("_")[1] for s in fWFK.split("/")[0].split("-")[:3] )
except Exception as  err:
    print (err)
#    exit()
    preline=""



if (refUC is not None) and (shiftUC is None): shiftUC=np.zeros(3)




bandstr=BS.BandStructure(fWAV=fWAV,fWFK=fWFK,fPOS=fPOS,Ecut=Ecutsym,IBstart=IBstart,IBend=IBend,kplist=kpoints,spinor=spinor,code=code,EF=EF,onlysym=onlysym)
bandstr.spacegroup.show(refUC=refUC,shiftUC=shiftUC,symmetries=symmetries)
if onlysym : 
#    print ("onlysym")
    exit()
    
#print (refUC,shiftUC)    



if refUC is None: refUC=np.eye(3)
if shiftUC is None: shiftUC=np.zeros(3)
open("irreptable-template","w").write(bandstr.spacegroup.str(refUC=refUC,shiftUC=shiftUC))

subbands={():bandstr}



if isymsep is not None :
   for isym in isymsep:
        print ("Separating by symmetry operation # ",isym)
        subbands={tuple(list(s_old)+[s_new]):sub for s_old,bands in subbands.items() for s_new,sub in bands.Separate(isym,degen_thresh=degen_thresh,groupKramers=groupKramers).items()}



if ZAK:
    for k in subbands:
        print ("eigenvalue {0}".format(k) )
        subbands[k].write_characters(degen_thresh=0.001,refUC=refUC,symmetries=symmetries)
        print("eigenvalue : #{0} \n Zak phases are : ".format(k) )
        ZAK=subbands[k].zakphase()
        for n,(z,gw,gc,lgw) in enumerate(zip(*ZAK)):
            print("   {n:4d}    {z:8.5f} pi gapwidth = {gap:8.4f} gapcenter = {cent:8.3f} localgap= {lg:8.4f}".format(n=n+1,z=z/np.pi,gap=gw,cent=gc,lg=lgw))


if WCC:
    for k in subbands:
        print ("eigenvalue {0}".format(k) )
#        subbands[k].write_characters(degen_thresh=0.001,refUC=refUC,symmetries=symmetries)
        wcc=subbands[k].wcc()
        print("eigenvalue : #{0} \n  WCC are : {1} \n sumWCC={2}".format(k,wcc,np.sum(wcc)%1) )

   
def short(x,nd=3):
    fmt="{{0:+.{0}f}}".format(nd) 
    if abs(x.imag)<10**(-nd): return fmt.format(x.real)
    if abs(x.real)<10**(-nd): return fmt.format(x.imag)+"j"
    return (short(x.real,nd)+short(1j*x.imag))
    


if writebands:
    bandstr.write_trace(degen_thresh=degen_thresh,refUC=refUC,shiftUC=shiftUC,symmetries=symmetries)
    for k,sub in subbands.items():
        if isymsep is not None: 
            print ("\n\n\n\n ################################################ \n\n\n next subspace:  ", " , ".join("{0}:{1}".format(s,short(ev)) for s,ev in zip(isymsep,k)))
        sub.write_characters(degen_thresh=degen_thresh,refUC=refUC,shiftUC=shiftUC,symmetries=symmetries,kpnames=kpnames,preline=preline,plotFile=plotFile)



if plotbands:
   for k,sub in subbands.items():
        if isymsep is not None: 
            print ("\n\n\n\n ################################################ \n\n\n next subspace:  ", " , ".join("{0}:{1}".format(s,short(ev)) for s,ev in zip(isymsep,k)))
            fname="bands-"+seedname+"-"+"-".join("{0}:{1}".format(s,short(ev)) for s,ev in zip(isymsep,k))+".dat"
            fname1="bands-sym-"+seedname+"-"+"-".join("{0}:{1}".format(s,short(ev)) for s,ev in zip(isymsep,k))+".dat"
        else:
            fname="bands-{0}.dat".format(seedname)
            fname1="bands-sym-{0}.dat".format(seedname)
        open(fname,"w").write(sub.write_bands())
        sub.write_trace_all(degen_thresh,fname=fname1)



