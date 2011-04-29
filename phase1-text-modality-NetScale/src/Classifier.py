import numpy, math, cPickle, sys
from linearutil import *
from numpy.random import shuffle

# 
def get_clserr(preds, true_labels):
    """
    get_clserr(preds, true_labels): -> CLSERR
    
    Compute the CLSERR given the list of predictions 
    and the true labels.
    """
    if len(true_labels)>0:
        res=0
        for pp in zip(preds, true_labels):
            if (pp[0]!=pp[1]):
                res+=1
        return res/float(len(preds))
    return 1

def get_loss(wx, true_labels):
    """
    get_loss(wx, true_labels): -> averaged loss

    Compute the L2 loss (only!) SVC given the 
    list of wx and the true labels.
    """
    if len(true_labels)>0:
        res=0
        for pp in zip(wx, true_labels):
            if pp[1] == 0.:
                lab = -1
            else:
                lab = 1
            res += max([0 , 1 - lab *  pp[0][0]])**2
        return res/float(len(wx))
    return 1

def TrainAndOptimizeClassifer(TrainingData, ValidationData, verbose, typ = 'normal'):
    """
    TrainAndOptimizeClassifer(TrainingData, ValidationData, verbose) -> Classifier_model

    Taking as input training data (stored in memory) and validation
    data (stored in memory), this function performs a line search to
    determine the best SVM C parameter and returns the best svm model
    (i.e. the one with the lowest validation CLSERR).
    """
     #linesearch looking for the best C
    MAXSTEPS=10
    STEPFACTOR=10.
    INITIALC=0.001

    Ccurrent = INITIALC
    Cstepfactor = STEPFACTOR
    Cnew = Ccurrent * Cstepfactor

    C_to_allstats = {}
    C_to_validloss = {}
    Cbest = None
    Models= {}

    TrainingProblem = problem(TrainingData[0],TrainingData[1])

    print >> sys.stderr, "Training on %d ex"% len(TrainingData[0])
    print >> sys.stderr, "Validating on %d ex"% len(ValidationData[0])

    print >> sys.stderr, "Performing line search to get the best C (%d steps)"% MAXSTEPS 
    while len(C_to_allstats) < MAXSTEPS:
        if Ccurrent not in C_to_allstats:
            # Compute the validation statistics for the current C
            param = '-c %f -s 1 -q -e 0.01'% Ccurrent
            m=train(TrainingProblem, param)
            preds, acc, probas = predict(ValidationData[0], ValidationData[1], m , '-b 0')
            C_to_allstats[Ccurrent] = get_clserr(preds, ValidationData[0])
            C_to_validloss[Ccurrent] = get_loss(probas, ValidationData[0])
            Models[Ccurrent]=m
        if Cnew not in C_to_allstats:
            # Compute the validation statistics for the next C
            param = '-c %f -s 1 -q -e 0.01'% Cnew
            m=train(TrainingProblem, param)
            preds, acc, probas = predict(ValidationData[0], ValidationData[1], m , '-b 0')
            C_to_allstats[Cnew] = get_clserr(preds, ValidationData[0])
            C_to_validloss[Cnew] = get_loss(probas, ValidationData[0])
            Models[Cnew]=m
          # If Cnew has a higher val clserr than Ccurrent, then continue stepping in this direction
        if C_to_allstats[Cnew] < C_to_allstats[Ccurrent]:
            if verbose: 
                print >> sys.stderr, "\tvalclserr[Cnew %f] = %f < valclserr[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew], Ccurrent, C_to_allstats[Ccurrent])
            if Cbest is None or C_to_allstats[Cnew] < C_to_allstats[Cbest]:
                Cbest = Cnew
                if verbose: 
                    print >> sys.stderr, "\tNEW BEST: Cbest <= %f, valclserr[Cbest] = %f" % (Cbest, C_to_allstats[Cbest])
            Ccurrent = Cnew
            Cnew *= Cstepfactor
            if verbose: 
                print >> sys.stderr, "\tPROCEED: Cstepfactor remains %f, Ccurrent is now %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)
        # Else, reverse the direction and reduce the step size by sqrt.
        else:
            if verbose: 
                print >> sys.stderr, "\tvalclserr[Cnew %f] = %f > valclserr[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew], Ccurrent, C_to_allstats[Ccurrent])
            if Cbest is None or C_to_allstats[Ccurrent] < C_to_allstats[Cbest]:
                Cbest = Ccurrent
                if verbose: 
                    print >> sys.stderr, "\tCbest <= %f, valclserr[Cbest] = %f" % (Cbest, C_to_allstats[Cbest])
            Cstepfactor = 1. / math.sqrt(Cstepfactor)
            Cnew = Ccurrent * Cstepfactor
            if verbose: 
                print >> sys.stderr, "\tREVERSE: Cstepfactor is now %f, Ccurrent remains %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)

    allC = C_to_allstats.keys()
    allC.sort()
    if verbose: 
        for C in allC:
            print >> sys.stderr, "\tvalclserr[C %f]/loss = %f,%f" % (C, C_to_allstats[C],C_to_validloss[C]),
            if C == Cbest: print >> sys.stderr, " *best*"
            else: print >> sys.stderr, ""
    else:
        print >> sys.stderr, "\tBestC %f with Validation CLSERR/loss = %f,%f" % (Cbest, C_to_allstats[Cbest],C_to_validloss[Cbest])
    
    if typ == 'Adistance':
        return C_to_allstats[Cbest]
    if typ == 'small':
        return Models[Cbest]
    if typ == 'normal':
        print >> sys.stderr, "Training on all data with the bestC: %f" % Cbest
        WholeProblem= problem(TrainingData[0]+ValidationData[0],TrainingData[1]+ValidationData[1])
        param = '-c %f -s 1 -q -e 0.01'% Cbest
        return train(TrainingProblem, param)
    if typ == 'normal2':
        print >> sys.stderr, "Training on all data with the bestC: %f" % Cbest
        WholeProblem= problem(TrainingData[0]+ValidationData[0],TrainingData[1]+ValidationData[1])
        param = '-c %f -s 1 -q -e 0.01'% Cbest
        return train(TrainingProblem, param),C_to_allstats[Cbest]
    if typ == 'normal3':
        return Cbest,C_to_allstats[Cbest]



def Classifier(model, TestData, prefix):
    """
    Classifier(model, TestData, prefix) -> None
    
    Taking as input a classifier model and test data (stored in
    memory), this function performs the model prediction for each test
    example and saves them into a file termed after the provided
    prefix.
    """
    # perform predictions
    print >> sys.stderr, "Testing on %d ex"% len(TestData[1])   
    preds, acc, probas = predict(TestData[0], TestData[1], model , '-b 0')
    
    # compute clserr (1 if no test labels given) and save predictions
    clserr=get_clserr(preds, TestData[0])
    print >> sys.stderr, "\tTest CLSERR = %f" % clserr    
    pf=open(prefix+'.predictions', 'w')
    for pp in preds:
        pf.write(str(pp)+'\n')
    pf.close()
    return clserr


def loadTrainDataset(task, labelFile, vectorFile, trainIDXFile,ones_fill=False):
    """
    loadTrainDataset(task, labelFile, vectorFile, trainIDXFile) -> Training Data, Validation Data

    Taking as input a large data set (given as a filename for labels
    -- labelFile, and a filename for feature vectors -- vectorFile), a
    task identifier (for OpenTable this is an integer between 0 and 4
    as they are 5 rating per example) and a list of index
    (trainIDXFile), this function returns a training set (containing
    all the examples listed in trainIDXFile) labeled with the rating
    of the identified task and a validation set containing the
    remaining examples also labeled.
    """
    TrainData=[[],[]]
    ValidationData=[[],[]]

    if type(labelFile) != list:
        labelFile = [labelFile]
        vectorFile = [vectorFile]
        trainIDXFile = [trainIDXFile]

    for tf in range(len(labelFile)):
        prob_y=svm_read_problem_labels(labelFile[tf])
        prob_x=svm_read_problem_vectors(vectorFile[tf],ones_fill)
        assert(len(prob_x)==len(prob_y))

        idx=dict()
        for line in open(trainIDXFile[tf]):
            idx[int(line.split()[0])]=True
        for i in range(len(prob_x)):
            if prob_y[i][task]>=0: # do not test on unrated examples (label=-1) only valid for the "noise" task)
                if i in idx:
                    TrainData[0] += [prob_y[i][task]]
                    TrainData[1] += [prob_x[i]]
                else:
                    ValidationData[0] += [prob_y[i][task]]
                    ValidationData[1] += [prob_x[i]]

    return TrainData, ValidationData


def loadTestDataset(task, labelFile, vectorFile,ones_fill=False):
    """
    loadTestDataset(task, labelFile, vectorFile) -> Test Data

    Taking as input a data set (given as a filename for labels --
    labelFile, and a filename for feature vectors -- vectorFile) and a
    task identifier (for OpenTable this is an integer between 0 and 4
    as they are 5 rating per example), this function returns a test
    set. If no labelFile is given, an unlabeled test set is retruned.
    """       
    prob_x=svm_read_problem_vectors(vectorFile,ones_fill)
    if labelFile:
        assert(task!=None)
        prob_y=svm_read_problem_labels(labelFile)
        labels=[]
        nprob_x=[]
        for i in zip(prob_y, prob_x):
            if i[0][task]>=0: # do not test on unrated examples (label=-1) only valid for the "noise" task)            
                labels += [i[0][task]]
                nprob_x+=[i[1]]
        return [labels, nprob_x]
    else:                        
        return [[], prob_x]


# parse command line arguments and call main function that will load
# data and build train, validation and test set, train and optimize a
# classifier using the train and the validation sets and save its test
# predictions into a file.
#
if __name__ == "__main__":

    if len(sys.argv) < 6:
        print "Usage:", sys.argv[0], " TaskIndex TrainVectors TrainLabels TrainIndices TestVectors [TestLabels]"
        sys.exit(-1)
        
    TaskIndex = int(sys.argv[1])
    TrainVectors = sys.argv[2]
    TrainLabels = sys.argv[3]
    TrainIndices = sys.argv[4]
    TestVectors=sys.argv[5]
    TestLabels=None
    if len(sys.argv) ==7:
        TestLabels=sys.argv[6]

    TrainingData, ValidationData = loadTrainDataset(TaskIndex, TrainLabels, TrainVectors, TrainIndices)
    best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, False)
    del TrainingData
    del ValidationData
       
    TestData = loadTestDataset(TaskIndex, TestLabels, TestVectors)
    Classifier(best_classifier, TestData, TestVectors.rpartition('/')[2].rpartition('.')[0]+'_task'+str(TaskIndex))
