#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N irace-classical
#$ -o out/
#$ -e out/
#$ -l h_vmem=8G
#$ -l h_rt=04:00:00

set -euo pipefail

# -------------------------
# Live logging you can tail
# -------------------------
mkdir -p out out_50
LOG="out_50/irace_${JOB_ID:-manual}_size${JOB_SIZE:-NA}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== IRACE JOB START ==="
echo "DATE: $(date)"
echo "JOB_ID: ${JOB_ID:-unknown}"
echo "HOST: $(hostname)"
echo "WORKDIR: $(pwd)"
echo "LOG: $LOG"

trap 'rc=$?; echo "=== IRACE JOB END ==="; echo "DATE: $(date)"; echo "EXIT_CODE: $rc"; exit $rc' EXIT

# -----------------------
# Environment
# -----------------------
source /home1/share/conda/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate r-irace

# Require JOB_SIZE
: "${JOB_SIZE:?Submit with: qsub -v JOB_SIZE=50 run_irace_final.sh}"

SIZE="${JOB_SIZE}"
INSTANCES="instances-list-${SIZE}.txt"

echo "=== IRACE JOB START ==="
echo "DATE: $(date)"
echo "JOB_ID: ${JOB_ID:-unknown}"
echo "HOST: $(hostname)"
echo "WORKDIR: $(pwd)"
echo "JOB_SIZE: ${SIZE}"
echo "INSTANCES FILE: ${INSTANCES}"
echo "R: $(Rscript -e 'cat(R.version.string)')"
echo "======================="


# -----------------------
# Prepare directories
# -----------------------
mkdir -p out "irace_out/${SIZE}"
cd "irace_out/${SIZE}"

# -----------------------
# Copy required files
# -----------------------
cp -f ../../scenario_template.txt ./scenario_template.txt
cp -f ../../parameters.txt        ./parameters.txt
cp -f ../../target-runner         ./target-runner
chmod +x ./target-runner
cp -f ../../main.jl               ./main.jl
cp -f "../../${INSTANCES}"        "./${INSTANCES}"

# -----------------------
# Generate scenario.txt
# -----------------------
sed "s|@INSTANCES@|${INSTANCES}|g" scenario_template.txt > scenario.txt

echo "Scenario trainInstancesFile:"
grep -n "trainInstancesFile" scenario.txt || true


# -----------------------
# Run irace (R interface)
# -----------------------
Rscript -e '
  library(irace)
  scenario <- readScenario("scenario.txt", scenario = defaultScenario())
  checkIraceScenario(scenario = scenario)
  irace_main(scenario = scenario)
'

echo "=== IRACE JOB END ==="
echo "DATE: $(date)"

