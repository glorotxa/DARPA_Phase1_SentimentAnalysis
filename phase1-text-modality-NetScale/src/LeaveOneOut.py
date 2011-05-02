import numpy, math, cPickle, sys
from linearutil import *
from Classifier import *
import cPickle
import os
import sys

def SampleStratifiedFolds(ListLabel,FoldsNumber):
    """
    This function, given a list of labels and a number of fold
    will return a list of list of index (one for each fold) corresponding
    to the stratified folds.
    """
    sorted_idx = numpy.argsort(ListLabel)
    kfolds = []
    for i in range(FoldsNumber):
        kfolds += [list(numpy.asarray(sorted_idx)[i::FoldsNumber])]
    return kfolds



#def EstimateC(DataPath,depth):
#    C = []
#    vldres = []
#    if depth!=0:
#        TrainVectors = DataPath + '_DL%s.vec'%(depth)
#    else:
#        TrainVectors = DataPath + '.vec'
#    TrainLabels = DataPath + '.lab'
#    Data = loadTestDataset(0, TrainLabels, TrainVectors)
#    lfolds = SampleStratifiedFolds(Data[0],4)
#    for i in range(4):
#        tst = [[],[]]
#        oth = [[],[]]
#        for k in range(len(Data[0])):
#            if k in lfolds[i]:
#                tst[0] += [Data[0][k]]
#                tst[1] += [Data[1][k]]
#            else:
#                oth[0] += [Data[0][k]]
#                oth[1] += [Data[1][k]]
#        bestC,valerr = TrainAndOptimizeClassifer(oth, tst, True, 'normal3')
#        vldres += [valerr]
#        C += [bestC]
#    return numpy.mean(C), numpy.std(C)


def LeaveOneOut(DataPath,listtotest):
    TrainVectors = DataPath + '_DL3.vec'
    TrainLabels = DataPath + '.lab'
    Data = loadTestDataset(0, TrainLabels, TrainVectors)
    res = []
    for i in listtotest:
        prob = problem(Data[0][:i]+Data[0][(i+1):],Data[1][:i]+Data[1][(i+1):])
        param = '-c 0.0001 -s 1 -q -e 0.01'
        model = train(prob, param)
        preds, acc, probas = predict(Data[0][i:(i+1)], Data[1][i:(i+1)], model , '-b 0')
        res+=[acc]
        print "Sample # %s current test accuracy: %s %"%(i,numpy.mean(res))
    return res


#def LeaveOneOutGeneric(C,DataPath,depth,listtotest):
#    if depth!=0:
#        TrainVectors = DataPath + '_DL%s.vec'%(depth)
#    else:
#        TrainVectors = DataPath + '.vec'
#    TrainLabels = DataPath + '.lab'
#    Data = loadTestDataset(0, TrainLabels, TrainVectors)
#    res = []
#    for i in listtotest:
#        prob = problem(Data[0][:i]+Data[0][(i+1):],Data[1][:i]+Data[1][(i+1):])
#        param = '-c %s -s 1 -q -e 0.01'%C
#        model = train(prob, param)
#        preds, acc, probas = predict(Data[0][i:(i+1)], Data[1][i:(i+1)], model , '-b 0')
#        res+=[acc]
#    return res

#def LeaveOneOutCheap(C,DataPath,depth,listtotest):
#    if depth!=0:
#        TrainVectors = DataPath + '_DL%s.vec'%(depth)
#    else:
#        TrainVectors = DataPath + '.vec'
#    TrainLabels = DataPath + '.lab'
#    Data = loadTestDataset(0, TrainLabels, TrainVectors)
#    res = []
#    n = len(Data[0])/20
#    for i in range(len(Data[0])/n):
#        prob = problem(Data[0][:n*i]+Data[0][n*(i+1):],Data[1][:n*i]+Data[1][n*(i+1):])
#        param = '-c %s -s 1 -q -e 0.01'%C
#        model = train(prob, param)
#        preds, acc, probas = predict(Data[0][n*i:n*(i+1)], Data[1][n*i:n*(i+1)], model , '-b 0')
#        res+=[acc]
#        print numpy.mean(res),numpy.std(res),i
#    return res


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print "Usage:", sys.argv[0], "DataPath"
        sys.exit(-1)
    
    DataPath = sys.argv[1]
    listtotest = list(numpy.arange(8000))
    res = LeaveOneOut(DataPath,listtotest)
    print "---------------------------------"
    print "Leave-one-out evaluation finished"
    print "TEST ACCURACY: ",numpy.mean(res),"%"
    
