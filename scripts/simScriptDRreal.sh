#!/bin/usr/env bash
nsims=2500
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
for data_name in "lalonde_cps" "lalonde_psid" "twins"
do
  sbatch  --export=data_name=$data_name ~/DRinference/scripts/simScriptDRreal.sbatch
done

