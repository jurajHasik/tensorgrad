#!/bin/bash

source $HOME/Software/miniconda3/bin/activate pytorch

ad=4
ch=48
j=1.0
for seed in 372197 75842 64185 9921462 74832 58284
do
echo "started J2=$j at $(date)" 
python -u variational.py -seed $seed -omp_cores 4 -D $ad -chi $ch -J1 0.0 -J2 $j > run_J10.0_J2$j-D$ad-chi${ch}-seed${seed}.out
done
