#!/bin/usr/env bash
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
for data_name in "ihdp" "lalonde_cps" "lalonde_psid" "twins" "acic2017_19" "acic2017_20" "acic2017_23" "acic2017_24"
  do
    sbatch  --export=data_name=$data_name ~/DRinference/scripts/simScriptDRrealxgboost.sbatch
  done
