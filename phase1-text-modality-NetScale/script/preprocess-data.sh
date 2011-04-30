#!/bin/sh
set -e
PROJECTROOT=`dirname $0`/..
DATADIR=$PROJECTROOT
FULLAMAZONDIR=__path/to/your/"unprocessed"/folder__
SMALLAMAZONDIR=__path/to/your/"processed_acl"/folder__



# Generate the data in our libsvm type files with the DICTSIZE
# most frequent features (Note if DICTSIZE=0, the dictionary contains all the features)
DICTSIZE=5000
python $PROJECTROOT/src/PreProcess.py $DATADIR/featDict.txt $DATADIR/preprocessed-fullamazon $DATADIR/preprocessed-smallamazon $DICTSIZE $FULLAMAZONDIR $SMALLAMAZONDIR
