#!/bin/sh
set -e
LAUNCHDIR=`pwd`
cd `dirname $0`
PROJECTROOT=..
DATADIR=$PROJECTROOT

export PYTHONPATH=$PROJECTROOT/DLmodel:$PYTHONPATH

python $PROJECTROOT/src/LeaveOneOut.py $PROJECTROOT/preprocessed-smallamazon 

cd $LAUNCHDIR

