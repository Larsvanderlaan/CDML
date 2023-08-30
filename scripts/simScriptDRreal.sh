#!/bin/usr/env bash
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
<<<<<<< HEAD
for data_name in "lalonde_cps" "lalonde_psid" "twins" #"acic2017_19" "acic2017_20" "acic2017_23" "acic2017_24" "ihdp"
=======
for data_name in "ihdp" "lalonde_cps" "lalonde_psid" "twins" "acic2017_19" "acic2017_20" "acic2017_23" "acic2017_24"
>>>>>>> 7201deb7813bbcf112d940fc24a6569bd0155542
do
  sbatch  --export=data_name=$data_name ~/DRinference/scripts/simScriptDRreal.sbatch
done

