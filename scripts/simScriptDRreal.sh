#!/bin/usr/env bash
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
for data_name in "acic2018_1000" "acic2018_2500" "acic2018_5000" "acic2018_10000" #"ihdp" "lalonde_cps" "lalonde_psid" "twins" "acic2017_17" "acic2017_18" "acic2017_19" "acic2017_20" "acic2017_21" "acic2017_22" "acic2017_23" "acic2017_24"
  do
    sbatch  --export=data_name=$data_name ~/DRinference/scripts/simScriptDRreal.sbatch
  done

