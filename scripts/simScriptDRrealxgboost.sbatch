#!/bin/bash

#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --output=Rout/par-%J.out
#SBATCH --error=Rout/par-%J.err
echo "LOADING R"
module load R
echo "R LOADED"
Rscript -e 'source("~/DRinference/scripts/R_setup.R"); source("~/DRinference/scripts/DRrealdataxgboost.R"); data_name = as.character(Sys.getenv("data_name")); do_real_data(data_name = data_name)'
