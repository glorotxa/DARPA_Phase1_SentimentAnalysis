#!/bin/sh
set -e
LAUNCHDIR=`pwd`
cd `dirname $0`
PROJECTROOT=..
DATADIR=$PROJECTROOT

export PYTHONPATH=$PROJECTROOT/DLmodel:$PYTHONPATH

THEANO_FLAGS=device=cpu,floatX=float32 python $PROJECTROOT/src/CreateRep.py $PROJECTROOT/DLmodel/Saved_Model/  $PROJECTROOT/preprocessed-smallamazon 

cd $LAUNCHDIR

