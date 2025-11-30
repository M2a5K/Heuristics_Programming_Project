#!/bin/bash
#$ -N search_experiments
#$ -j y
#$ -o search_experiments.log
#$ -l h_rt=00:15:00

# Submit with: qsub run_11b.sh

# Load Julia module (adjust version as needed for your cluster)
# module load julia/1.11.0

# Set Julia to use single thread (required by the script)
export JULIA_NUM_THREADS=1

echo "MHLibDemos directory: $MHLIBDEMOS_DIR"

# Run Julia with the experiment script using absolute path
julia --project="$MHLIBDEMOS_DIR" -e "
using Pkg
Pkg.instantiate()
using MHLibDemos
run11b()
"

echo "Search experiments completed"
