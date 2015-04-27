#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=1:00:00
#PBS -l mem=64GB
#PBS -N Jake

module load lua
module load cuda/6.5.12
module load torch-deps/7

#
export TORCHROOT=/scratch/jz1672/torch/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/pkg/cuda/6.5/cuda/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/pkg/cuda/6.5/cuda/lib64
export MODULEPATH=$MODULEPATH:/usr/local/etc/modulefiles
#
#
export PATH=$TORCHROOT/install/bin:$PATH
export LD_LIBRARY_PATH=$TORCHROOT/install/lib/lua/5.1:/home/jz1672/torch/torch-distro/install/lib:$LD_LIBRARY_PATH  # Added automatically by torch-dist
export DYLD_LIBRARY_PATH=$TORCHROOT/install/lib:$DYLD_LIBRARY_PATH  # Added automatically by torch-dist
export LIBRARY_PATH=$TORCHROOT/install/lib:$LIBRARY_PATH

#cd ~
#mkdir -p Jake
#cd Jake
#cp -r /home/jz1672/submit .
#cd submit

/scratch/jz1672/torch/install/bin/th main.lua
