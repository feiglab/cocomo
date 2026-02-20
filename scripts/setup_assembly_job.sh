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

default_nsteps=5000000
default_maxrun=10

[[ -n "${cfg_nsteps:-}" ]] && default_nsteps=$cfg_nsteps
[[ -n "${cfg_maxrun:-}" ]] && default_maxrun=$cfg_maxrun

nsteps=${nsteps_arg:-$default_nsteps}
maxrun=${maxrun_arg:-$default_maxrun}

cur_base=$(basename -- "$PWD")               # e.g. set1
parent=$(basename -- "$(dirname -- "$PWD")") # e.g. ht

if [[ "$cur_base" != set* ]]; then
  echo "error: expected current dir name like 'setN', got '$cur_base' (PWD=$PWD)" >&2
  exit 1
fi

setnum=${cur_base#set}                       # e.g. 1
if [[ -z "$setnum" || ! "$setnum" =~ ^[0-9]+$ ]]; then
  echo "error: expected numeric set number in '$cur_base' (PWD=$PWD)" >&2
  exit 1
fi

tag="AS.${parent}_${setnum}"

dir=$(pwd)
dir_esc=$(printf '%s' "$dir" | sed 's/[\/&|\\]/\\&/g')

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
template=${script_dir}/job.assembly.slurm.template

if [[ ! -r $template ]]; then
   echo "cannot find template"
   exit 1
fi

if [[ ! -r "$maxrun" ]]; then
   echo "$maxrun" > maxrun
fi

if [[ ! -r "last" ]]; then
   echo 0 > "last"
fi

sed \
  -e "s|DIR|$dir_esc|g" \
  -e "s/TAG/$tag/g" \
  -e "s/QUEUE/$queue/g" \
  -e "s/NSTEPS/$nsteps/g" \
  "$template" > "job.prod.slurm"

