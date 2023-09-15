#!/bin/usr/env bash
nsims=2500
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
for n in 100 250 500 750 1000 2000 3000 4000 5000 7500 9000
do
  for const in 2
      do
    for misp in 1
    do
    sbatch  --export=n=$n,const=$const ~/DRinference/scripts/simScriptDRkernel.sbatch
    done
  done
done
