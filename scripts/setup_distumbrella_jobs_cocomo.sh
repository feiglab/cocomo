#!/bin/bash
set -euo pipefail

queue_arg=${1:-}
nsteps_arg=${2:-}
maxrun_arg=${3:-}
workdir_arg=${4:-}

if [[ -n "$workdir_arg" && -d "$workdir_arg" ]]; then
    cd -- "$workdir_arg"
fi

# --- read defaults from config (if present) ---
default_queue=all
cfg_nsteps=""
cfg_maxrun=""
cfg_queue=""

if [[ -r config ]]; then
    cfg_nsteps=$(
        awk '$1=="nsteps" && NF>=2 { print $2; exit }' config
    )
    cfg_maxrun=$(
        awk '$1=="maxrun" && NF>=2 { print $2; exit }' config
    )
    cfg_queue=$(
        awk '$1=="queue" && NF>=2 { print $2; exit }' config
    )
fi

[[ -n "${cfg_queue:-}" ]] && default_queue=$cfg_queue
queue=${queue_arg:-$default_queue}

default_nsteps=10000000
default_maxrun=1
tag=CO

[[ -n "${cfg_nsteps:-}" ]] && default_nsteps=$cfg_nsteps
[[ -n "${cfg_maxrun:-}" ]] && default_maxrun=$cfg_maxrun

nsteps=${nsteps_arg:-$default_nsteps}
maxrun=${maxrun_arg:-$default_maxrun}

[[ -r tag ]] && tag=$(<tag)

dir=$(pwd)
dir_esc=$(printf '%s' "$dir" | sed 's/[\/&|\\]/\\&/g')

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
template=${script_dir}/job.distumbrella.cocomo.slurm.template

if [[ ! -r $template ]]; then
   echo "cannot find template"
   exit 1
fi

if [[ ! -r "$maxrun" ]]; then
   echo "$maxrun" > maxrun
fi

for n in run_*; do
  biasval=${n#run_}
  if [[ ! -r "$n/last" ]]; then
     echo 0 > "$n/last"
  fi

  sed \
    -e "s/BIAS/$biasval/g" \
    -e "s/BDIR/$n/g" \
    -e "s|DIR|$dir_esc|g" \
    -e "s/TAG/$tag/g" \
    -e "s/QUEUE/$queue/g" \
    -e "s/NSTEPS/$nsteps/g" \
    "$template" > "$n/job.prodbias.slurm"
done

