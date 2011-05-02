import cPickle
import os
import sys

import numpy
from exp_scripts.DARPAscript import createvecfile

        
def CreateRep(ModelPath,DataPrefix):
    createvecfile(ModelPath,DataPrefix+ '.vec',3,DataPrefix+ '_DL3.vec')
    createvecfile(ModelPath,DataPrefix+ '.vec',1,DataPrefix + '_DL1.vec')


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print "Usage:", sys.argv[0], "ModelPath DataPrefix"
        sys.exit(-1)
    
    ModelPath = sys.argv[1]
    DataPrefix = sys.argv[2]
    CreateRep ( ModelPath,DataPrefix )

