import cPickle, math, os, os.path, sys, time

import scipy

try:
    from ANN import *
except ImportError:
    from deepANN.ANN import *

# TRAINFUNC is a handle to the model's training function. It is a global
# because it is connected to internal state in the Model.
TRAINFUNC       = None


def read_hyperparameters_file(path):
    """
    This function reads the hyperparameter file of the experiment and 
    return it as a dictionnary.
    """ 
    f = open(path,'r')
    paramdict = {}
    line = f.readline()
    while line != '':
        paramval = line[:-1].split('=')
        if len(paramval)==2 and line[0]!='#':
            param, value = paramval[0],paramval[1]
            idx = -1
            while param[idx] == ' ':
                idx -=1
            if idx != -1:
                param = param[:idx+1]
            idx = 0
            while value[idx] == ' ':
                idx +=1
            value = value[idx:]
            paramdict.update({param:eval(value)})
        line = f.readline()
    print >> sys.stderr , "Hyperparameters dictionnary : ", paramdict
    return paramdict

def vectosparsemat(path,NBDIMS):
    """
    This function converts the unlabeled training data into a scipy
    sparse matrix and returns it.
    """
    print >> sys.stderr , "Read and converting data file: %s to a sparse matrix"%path 
    # We first count the number of line in the file
    f = open(path, 'r')
    i = f.readline()
    ct = 0
    while i!='':
        ct+=1
        i = f.readline()
    f.close()
    # We allocate and fill the sparse matrix as a lil_matrix for efficiency.
    NBEX = ct
    train = scipy.sparse.lil_matrix((NBEX,NBDIMS))
    f = open(path, 'r')
    i = f.readline()
    ct = 0
    next_print_percent = 0.1
    while i !='':
        if ct / float(NBEX) > next_print_percent:
            print >> sys.stderr , "\tRead %s %s of file"%(next_print_percent*100,'%')
            next_print_percent += 0.1
        i = i[:-1]
        i = list(i.split(' '))
        for j in i:
            if j!='':
                idx,dum,val = j.partition(':')
                train[ct,int(idx)] = 1
        i = f.readline()
        ct += 1
    print >> sys.stderr , "Data converted" 
    # We return a csr matrix for efficiency 
    # because we will later shuffle the rows.
    return train.tocsr()

def createdensebatch(spmat,size,batchnumber):
    """ 
    This function creates and returns a dense matrix corresponding to the 
    'batchnumber'_th slice of length 'size' of the sparse data matrix.
    If the last slice is smaller that size, the matrix is zero padded.
    The function also return the real number of examples present in the batch. 
    """
    NB_DENSE = int(numpy.ceil(spmat.shape[0] / float(size)))
    assert batchnumber>=0 and batchnumber<NB_DENSE 
    realsize = size
    if batchnumber< NB_DENSE-1:
        batch = numpy.asarray(spmat[size*batchnumber:size*(batchnumber+1),:].toarray(),dtype=theano.config.floatX)
    else:
        batch = numpy.asarray(spmat[size*batchnumber:,:].toarray(),dtype=theano.config.floatX)
        realsize = batch.shape[0]
        if batch.shape[0] < size:
            batch = numpy.concatenate([batch,numpy.zeros((size-batch.shape[0],batch.shape[1]),dtype=theano.config.floatX)])
    return batch,realsize
 

def rebuildunsup(model,depth,PARAMS,noise_lvl,train):
    """
    Modify the global TRAINFUNC.
    """
    global TRAINFUNC
    model.ModeAux(depth+1,update_type='special',noise_lvl=noise_lvl,lr=PARAMS['lr'][depth])
    if depth > 0:
        givens = {}
        index = T.lscalar()
        givens.update({model.inp : train[index*PARAMS['batchsize']:(index+1)*PARAMS['batchsize']]})
        givens.update({model.auxtarget : model.layers[depth-1].out})
        TRAINFUNC = theano.function([index],model.cost, updates = model.updates, givens = givens)
    else:
        TRAINFUNC,n = model.trainfunctionbatch(train,None,train, batchsize=PARAMS['batchsize'])


def createvecfile(PathLoad,PathData,depth,OutFile,BATCH_MAX = 500):
    """
    This function builds an 'OutFile' *.vec file corresponding to 'PathData' taken at 
    the layer 'depth' of the 'PathLoad' model.
    """
    print >> sys.stderr, "Creating vec file %s (model=%s, depth=%d, datafiles=%s)..." % (repr(OutFile), repr(PathLoad),depth,PathData)
    PARAMS = read_hyperparameters_file(PathLoad+'/DARPA.conf')
    NOISE = ['binomial_NLP'] + ['binomial'] * (PARAMS['depth']-1)
    model=SDAE(numpy.random,RandomStreams(PARAMS['seed']),PARAMS['depth'],True,act='rectifier',n_hid=PARAMS['n_hid'],n_out=5,sparsity=PARAMS['activation_regularization_coeff'],\
            regularization=PARAMS['weight_regularization_coeff'] + [0.0], wdreg = 'l2', spreg = 'l1', n_inp=PARAMS['ninputs'],noise=NOISE,tie=True) 
    model.load(PathLoad+'/depth%s'%depth)
    model.ModeSup(depth,depth,update_type='global',lr=0)
    outputs = [model.layers[depth-1].out]
    func = theano.function([model.inp],outputs)

    full_train = vectosparsemat(PathData,model.layers[0].W.value.shape[0])
    NB_BATCHS = int(numpy.ceil(full_train.shape[0] / float(BATCH_MAX)))

    f = open(OutFile,'w')

    for i in range(NB_BATCHS):
        if i < NB_BATCHS-1:
            rep = func(numpy.asarray(full_train[BATCH_MAX*i:BATCH_MAX*(i+1),:].toarray(),dtype=theano.config.floatX))[0]
        else:
            rep = func(numpy.asarray(full_train[BATCH_MAX*i:,:].toarray(),dtype=theano.config.floatX))[0]
        textr = ''
        for l in range(rep.shape[0]):
            idx = rep[l,:].nonzero()[0]
            for j,v in zip(idx,rep[l,idx]):
                textr += '%s:%s '%(j,v)
            textr += '\n'
        f.write(textr)
    f.close()
    print >> sys.stderr, "...done creating vec files"



def OpenTableSDAEexp(ConfigFile = None, SavePath = '.'):
    """
    This script launch a SDAE experiment, training in a greedy layer
    wise fashion.  The hidden layer activation function is the
    rectifier activation (i.e. max(0,y)). The reconstruction
    activation function is the sigmoid. The reconstruction cost is the
    cross-entropy. From one layer to the next we need to scale the
    parameters in order to ensure that the representation is in the
    interval [0,1].  The noise of the input layer is a salt and pepper
    noise ('binomial_NLP'), for deeper layers it is a zero masking
    noise (binomial).
    """
    # Load the hyperparameters from the .conf file
    if ConfigFile != None:
        PARAMS = read_hyperparameters_file(ConfigFile)
    else:
        PARAMS = read_hyperparameters_file('DARPA.conf')
    numpy.random.seed(PARAMS['seed'])

    # Load the entire training data and shuffle it
    full_train = vectosparsemat(PARAMS['path_data'],PARAMS['ninputs'])
    full_train = full_train[numpy.random.permutation(full_train.shape[0]),:]
    NB_DENSE = int(numpy.ceil(full_train.shape[0] / float(PARAMS['dense_size'])))
    
    # Create the dense batch shared variable
    train = theano.shared(createdensebatch(full_train,PARAMS['dense_size'],0)[0])

    # Create the model
    NOISE = ['binomial_NLP'] + ['binomial'] * (PARAMS['depth']-1)
    model=SDAE(numpy.random,RandomStreams(PARAMS['seed']),PARAMS['depth'],True,act='rectifier',n_hid=PARAMS['n_hid'],n_out=5,sparsity=PARAMS['activation_regularization_coeff'],\
            regularization=PARAMS['weight_regularization_coeff'] + [0.0], wdreg = 'l2', spreg = 'l1', n_inp=PARAMS['ninputs'],noise=NOISE,tie=True)
    if 'rld' in PARAMS.keys():
        model.load(PARAMS['rld'])
        begin = PARAMS['begin']
    else:
        begin = 0
    # Train the model in a greedy layer-wise fashion
    for depth in xrange(begin,PARAMS['depth']):
        print >> sys.stderr, 'BEGIN DEPTH %s...' %(depth+1)
        # Define the size of the DAE input
        if depth == 0:
            n_aux = PARAMS['ninputs']
        else:
            n_aux = model.layers[depth-1].n_out
        # Create the reconstrution layer
        model.depth_max = model.depth_max+1
        if model.auxlayer != None:
            del model.auxlayer.W
            del model.auxlayer.b
        if depth==0:
            model.auxiliary(init=1,auxdepth=-PARAMS['depth']+depth+1, auxn_out=n_aux)
        else:
            model.reconstruction_cost = 'quadratic'
            model.reconstruction_cost_fn = quadratic_cost
            model.auxiliary(init=1,auxdepth=-PARAMS['depth']+depth+1, auxact='softplus',auxn_out=n_aux)
        # Build the training function
        rebuildunsup(model,depth,PARAMS,PARAMS['noise_lvl'][depth],train) #build graph with noise for actual training
        # Train the current DAE
        for epoch in range(PARAMS['nepochs'][depth]):
            # Load sequentially dense batches of the training data
            for batchnb in range(NB_DENSE):
                train.container.value[:], realsize = createdensebatch(full_train,PARAMS['dense_size'],batchnb) 
                reconstruction_error_batch = 0
                update_count = 0
                for j in range(realsize/PARAMS['batchsize']):
                    # Update function 
                    reconstruction_error_batch += TRAINFUNC(j)
                    update_count += 1
                print >> sys.stderr, "\t\tAt depth %d, epoch %d, finished training over batch %s" % (depth+1, epoch+1, batchnb+1)
                print >> sys.stderr, "\t\tMean reconstruction error %s" % (reconstruction_error_batch/float(update_count))
            print >> sys.stderr, '...finished training epoch #%s' % (epoch+1)
        # Save the final model
        if not os.path.isdir(SavePath):
            os.mkdir(SavePath)
        modeldir = os.path.join(SavePath, 'depth%s' % (depth+1))
        if not os.path.isdir(modeldir):
            os.mkdir(modeldir)
        model.save(modeldir)
        print >> sys.stderr, '...DONE DEPTH %s' % (depth+1)
    

if __name__=='__main__':
    OpenTableSDAEexp()

