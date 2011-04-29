#!/bin/sh
set -e

PROJECTROOT=`dirname $0`/..
cd $PROJECTROOT/lib/liblinear/python

# make the C++
make

# set up symlinks
mkdir -p ../install/bin/
cd ../install/bin/
ln -s ../../liblinear/run_all
ln -s ../../liblinear/train
ln -s ../../liblinear/predict

# back to the root
cd ../..
