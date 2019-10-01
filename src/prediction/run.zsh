#!/bin/bash

#!/usr/bin/env zsh

### Job name
#SBATCH -J run
#SBATCH -e ./run.txt
#SBATCH -o ./run.txt

### Time your job needs to execute, e. g. 15 min 30 sec
#SBATCH -t 10:00:00

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem=50G --gres=gpu:1 --partition=c18g -A rwth0429

# PATH
export PATH=/home/rs619065/local/bin:$PATH
export PATH=/home/rs619065/.local/bin:$PATH
export PATH=/home/rs619065/local/bamtools/bin:$PATH
export PATH=/usr/local_rwth/sw/cuda/9.0.176/bin:$PATH
export PATH=/home/rs619065/perl5/bin:$PATH
export PATH=/home/rs619065/opt/cellranger-atac-1.1.0:$PATH

################################################################
# LIBRARYPATH
export LD_LIBRARY_PATH=/home/rs619065/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/rs619065/local/bamtools/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local_rwth/sw/cuda/9.0.176/lib64:$LD_LIBRARY_PATH

export R_LIBS_USER=$R_LIBS_USER:/home/rs619065/local/lib64/R/library
export PERL5LIB=/home/rs619065/perl5/lib/5.26.1:$PERL5LIB
export PERL5LIB=/home/rs619065/perl5/lib/perl5:$PERL5LIB

source ~/miniconda2/bin/activate py36env

python predict.py --fold 0
python predict.py --fold 1
python predict.py --fold 2
python predict.py --fold 3
python predict.py --fold 4
