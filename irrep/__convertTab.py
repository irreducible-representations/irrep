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


from  .__irreptable import IrrepTable as IRT



for sg in range(300):
  try:
    IRT(SGnumber=sg,spinor=True,fromUser=False).save4user()
    IRT(SGnumber=sg,spinor=False,fromUser=False).save4user()
  except Exception as err:
    print (err)
    pass