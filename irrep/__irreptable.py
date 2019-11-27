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


import numpy as np
import os,sys
import copy
from .__aux import str2bool,str_,str2list_space





class symop_table():


   def __init__(self,line,fromUser=False):
       if fromUser:
           self.__init__fromUser(line)
           return
       numbers=line.split()
       self.R=np.array(numbers[:9],dtype=int).reshape(3,3)
       self.t=np.array(numbers[9:12],dtype=float)
       self.S=(np.array(numbers[12::2],dtype=float)*np.exp(1j*np.pi*np.array(numbers[13::2],dtype=float))).reshape(2,2)


   def __init__fromUser(self,line):
       numbers=line.split()
       self.R=np.array(numbers[:9],dtype=int).reshape(3,3)
       self.t=np.array(numbers[9:12],dtype=float)
       if len(numbers)>12:
           self.S=(np.array(numbers[12:16],dtype=float)*np.exp(1j*np.pi*np.array(numbers[16:20],dtype=float))).reshape(2,2)
       else:
           self.S=np.eye(2)

       
   def str(self,spinor=True):
       return ("   ".join(" ".join(str(x) for x in r) for r in self.R)+"     "+" ".join(str_(x) for x in self.t)+ 
           (("      "+"    ".join("  ".join(str_(x) for x in X) for X in (np.abs(self.S.reshape(-1)), np.angle(self.S.reshape(-1))/np.pi) ))  if spinor else "") ) 




class charfunction():

    def __init__(self,abcde):
        self.abcde=copy.deepcopy(abcde)

    def __call__(self,u=0,v=0,w=0):
       return  sum(  aaa[0]*np.exp(1j*np.pi*(sum(a*u for a,u in zip(aaa[1:],(1,u,v,w)) )) ) for aaa in self.abcde) 


class KP():

    def __init__(self,name=None,k=None,isym=None,line=None):
        if line is not None:
            line_=line.split(':')
            if line_[0].split()[0]!='kpoint' : raise ValueError 
            self.name=line_[0].split()[1]
            self.k=np.array(line_[1].split(),dtype=float)
            self.isym=  str2list_space(line_[2])      #[ int(x) for x in line_[2].split() ]  #
        else:
            self.name=name
            self.k=k
            self.isym=isym


        
    def __eq__(self,other):
        if self.name != other.name: return False
        if np.linalg.norm(self.k-other.k)>1e-8 : return False
        if self.isym != other.isym: return False
        return True
        
    def show(self):
        return  "{0} : {1}  symmetries : {2}".format(self.name,self.k,self.isym)


    def str(self):
        return  "{0} : {1}  : {2}".format(self.name," ".join(str(x) for x in self.k)," ".join(str(x) for x in sorted(self.isym)) )

class irrep():
    def __init__(self,f=None,nsym_group=None,line=None,KP=None):
        if KP is not None:
            self.__init__user(line,KP)
            return
        s=f.readline().split()
#        print (s)
        self.k=np.array(s[:3],dtype=float)
        self.hasRkmk=True if s[3]=="1" else "0" if s[3]==0 else None
        self.name=s[4]
        self.kpname=s[7]
        self.dim=int(s[5])
        self.nsym=int(int(s[6])/2)
        self.reality=int(s[8])
        self.characters={}
        self.hasuvw=False
        for isym in range(1,nsym_group+1):
            ism,issym=[int(x) for x in f.readline().split()]
            assert ism==isym
#            print ("ism,issym",ism,issym)
            if issym==0:
                continue
            elif issym!=1:
                raise RuntimeError("issym should be 0 or 1, <{0}> found".format(issym) )
            abcde=[]
            hasuvw=[]
            for i in range(self.dim):
              for j in range(self.dim):
                l1,l2=[f.readline() for k in range(2)]
                if i!=j: continue  # we need only diagonal elements
                l1=l1.strip()
                if l1=="1":
                    hasuvw.append(False)
                elif l1=="2":
                    hasuvw.append(True)
                else: 
                    raise RuntimeError("hasuvw should be 1 or 2. <{0}> found".format(l1))
                abcde.append(np.array(l2.split(),dtype=float))
            if any(hasuvw): self.hasuvw=True
            if isym<=nsym_group/2:
                self.characters[isym]=charfunction(abcde)
        if not self.hasuvw:
            self.characters={isym:self.characters[isym]() for isym in self.characters}
#        print ("characters are:",self.characters)
        assert len(self.characters)==self.nsym

    def __init__user(self,line,KP):
#        print ("reading irrep line <{0}> for KP=<{1}> ".format(line,KP.str()))
        self.k=KP.k
        self.kpname=KP.name
        line=line.split()
        self.name=line[0]
        self.dim=int(line[1])
        self.nsym=len(KP.isym)
        self.reality=(len(line[2:])==self.nsym)
        ch=np.array(line[2:2+self.nsym],dtype=float)
        if (not self.reality): 
            ch=ch*np.exp(1.j*np.pi*np.array(line[2+self.nsym:2+2*self.nsym],dtype=float))
        self.characters={KP.isym[i]:ch[i] for i in range(self.nsym)}
#        print ("the irrep {0}  ch= {1}".format(self.name,self.characters))

         
    def show(self):
        print (self.kpname,self.name,self.dim,self.reality)
        
    def str(self):
#        print(self.characters)
        ch=np.array([self.characters[isym] for isym in sorted(self.characters)])
        if np.abs(np.imag(ch)).max()>1e-6:
            str_ch="   "+"  ".join(str_(x) for x in np.abs(ch))
            str_ch+="   "+"  ".join(str_(x) for x in np.angle(ch)/np.pi)
        else:
            str_ch="   "+"  ".join(str_(x) for x in np.real(ch))
        return self.name+" {} ".format(self.dim)+str_ch 
        

class IrrepTable():
    
    def __init__(self,SGnumber,spinor,fromUser=True,name=None):
        if fromUser: 
           self.__init__user(SGnumber,spinor,name)
           return
        self.number=SGnumber
        f=open(os.path.dirname(os.path.realpath(__file__)) +"/TablesIrrepsLittleGroup/TabIrrepLittle_{0}.txt".format(self.number),"r")
        self.nsym,self.name=f.readline().split()
        self.spinor=spinor
        self.nsym=int(self.nsym)
        self.symmetries=[symop_table(f.readline())    for i in range(self.nsym)]
        assert (f.readline().strip()=="#")
        self.NK=int(f.readline())
        self.irreps=[]
        try:
           while True:
               self.irreps.append(irrep(f=f,nsym_group=self.nsym) )
#               print ("irrep appended:")
#               self.irreps[-1].show()
               f.readline()
        except  EOFError :
            pass
        except IndexError as err:
#            print (err)
            pass
            
        if self.spinor: 
            self.irreps = [s for s in self.irreps if s.name.startswith("-")]
        else  : 
            self.irreps = [s for s in self.irreps if not s.name.startswith("-")]
            
        self.nsym=int(self.nsym/2)
        self.symmetries=self.symmetries[0: self.nsym]


    def show(self):
        for i,s in enumerate(self.symmetries) :
            print (i+1,"\n",s.R,"\n",s.t,"\n",s.S,"\n\n")
        for irr in self.irreps:
            irr.show()
            
    def save4user(self,name=None):
        if name is None:
            name="irreptables/irreps-SG={SG}-{spinor}.dat".format(SG=self.number,spinor="spin" if self.spinor else "scal")
        fout=open(name,"w")
        fout.write("SG={SG}\n name={name} \n nsym= {nsym}\n spinor={spinor}\n".format(SG=self.number,name=self.name,nsym=self.nsym,spinor=self.spinor) )
        fout.write("symmetries=\n"+"\n".join(s.str(self.spinor) for s in self.symmetries)+"\n\n" )
        
        kpoints={}
        
        for irr in self.irreps:
          if not irr.hasuvw:
            kp=KP(irr.kpname,irr.k,set(irr.characters.keys()))
            if len(set([0.123,0.313,1.123,0.877,0.427,0.246,0.687]).intersection(list(kp.k)) )==0 :
             try:
                assert(kpoints[kp.name]==kp)
             except KeyError:
                kpoints[kp.name]=kp

        
        for kp in kpoints.values():
            fout.write("\n kpoint  "+kp.str()+"\n")
            for irr in self.irreps:
              if irr.kpname==kp.name:
                fout.write(irr.str()+"\n")
        fout.close()



            
        
    def __init__user(self,SG,spinor,name):
        self.number=SG
        self.spinor=spinor
        if name is None:
            name="{root}/irreptables/irreps-SG={SG}-{spinor}.dat".format(SG=self.number,spinor="spin" if self.spinor else "scal",root=os.path.dirname(__file__))
        print ("reading from a user-defined irrep table <{0}>".format(name) ) 
            
        lines=open(name).readlines()[-1::-1]
        while len(lines)>0:
            l=lines.pop().strip().split("=")
#            print (l,l[0].lower())
            if    l[0].lower()=='SG' : assert(int(l[1])==SG)
            elif  l[0].lower()=='name' : self.name=l[1]
            elif  l[0].lower()=='nsym' : self.nsym=int(l[1])
            elif  l[0].lower()=='spinor' : assert(str2bool(l[1])==self.spinor)
            elif  l[0].lower()=='symmetries':
                print ('reading symmetries')
                self.symmetries=[]
                while len(self.symmetries)<self.nsym :
                    l=lines.pop()
#                    print (l)
                    try: 
                       self.symmetries.append( symop_table(l,fromUser=True) )
                    except Exception as err:
                       print (err)
                       pass
                break
        
#        print("symmetries are:\n"+"\n".join(s.str() for s in self.symmetries) )


        self.irreps=[]
        while len(lines)>0:
            l=lines.pop().strip()
            try:
                kp=KP(line=l)
#                print ("kpoint successfully read:",kp.str())
            except Exception as err:
#                print( "error while reading k-point <{0}>".format(l),err)
                try:
                    self.irreps.append(irrep(line=l,KP=kp))
                except Exception as err:
#                    print ("error while reading irrep <{0}>".format(l), err)
                    pass
            


