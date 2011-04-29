import cPickle
import os
import sys

import numpy

from exp_scripts.DARPAscript import OpenTableSDAEexp, createvecfile

def PreTrain(  DataPrefix, ModelPath, Seed ):
    numpy.random.seed(Seed)
    # learn the model
    OpenTableSDAEexp(ModelPath+'DARPA.conf',ModelPath)
    
if __name__ == '__main__':
    if len(sys.argv) <= 3:
        print "Usage:", sys.argv[0], "DataPrefix ModelPath Seed"
        sys.exit(-1)
    
    DataPrefix = sys.argv[1]
    ModelPath = sys.argv[2]
    Seed = int(sys.argv[3])
    PreTrain ( DataPrefix, ModelPath, Seed )
