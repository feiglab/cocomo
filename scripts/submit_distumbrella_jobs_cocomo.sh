#!/bin/bash
set -euo pipefail

if [[ -n "${1:-}" && -d "$1" ]]; then
    cd -- "$1"
fi

for n in run_*; do
    if [[ ! -r $n/running ]]; then
      sbatch $n/job.*.slurm
    fi
done
