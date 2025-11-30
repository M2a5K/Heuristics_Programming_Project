#!/bin/bash
#$ -N construction_experiments
#$ -j y
#$ -o construction_experiments.log
#$ -l h_rt=00:15:00
#$ -cwd

# Submit with: qsub run_11a.sh

# Set Julia to use single thread
export JULIA_NUM_THREADS=1

echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Starting Julia script test/run_11a.jl ..."

julia --project=. test/run_11a.jl

echo "Construction experiments completed"
