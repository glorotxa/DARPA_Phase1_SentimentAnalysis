#!/bin/sh
set -e
LAUNCHDIR=`pwd`
cd `dirname $0`
PROJECTROOT=..
DATADIR=$PROJECTROOT

export PYTHONPATH=$PROJECTROOT/DLmodel:$PYTHONPATH

SEED=777

THEANO_FLAGS=device=cpu,floatX=float32 python $PROJECTROOT/src/PreTrain.py $PROJECTROOT/preprocessed-fullamazon $PROJECTROOT/DLmodel/Saved_Model/ $SEED

cd $LAUNCHDIR

