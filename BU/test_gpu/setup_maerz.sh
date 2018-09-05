#!/bin/bash

LCG_RELEASE=93

if uname -a | grep ekpdeepthought
then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_${LCG_RELEASE}/x86_64-ubuntu1604-gcc54-dbg/setup.sh
    #pip install --user tensorflow-gpu==1.4.0
    export PATH=/usr/local/cuda-8.0/bin/:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES="3"
else
    source /cvmfs/sft.cern.ch/lcg/views/LCG_${LCG_RELEASE}/x86_64-slc6-gcc62-opt/setup.sh
fi
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH