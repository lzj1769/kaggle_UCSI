#!/bin/bash

################################################################
# PATH
export PATH=/usr/bin:$PATH
export PATH=/usr/local_host/bin:$PATH
export PATH=/home/rs619065/local/bin:$PATH
export PATH=/home/rs619065/.local/bin:$PATH
export PATH=/home/rs619065/local/bamtools/bin:$PATH
export PATH=/home/rs619065/perl5/bin:$PATH

################################################################
# LIBRARYPATH
export LD_LIBRARY_PATH=/home/rs619065/local/lib:$LD_LIBRARY_PATH

source ~/miniconda2/bin/activate py36env

python train.py --model $1 --fold $2 --batch-size $3