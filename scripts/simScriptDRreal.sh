#!/bin/usr/env bash
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
for data_name in "acic2017" "ihdp" "lalonde_cps" "lalonde_psid" "twins"
do
  sbatch  --export=data_name=$data_name ~/DRinference/scripts/simScriptDRreal.sbatch
done

