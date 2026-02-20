#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <nsteps>" >&2
  exit 2
}

nstep="${1:-}"
[[ -n "$nstep" ]] || usage

tstep=0.03
gamma=1.0

# Read last (default 0)
last=0
if [[ -r "last" ]]; then
  last="$(< "last")"
fi
if ! [[ "$last" =~ ^[0-9]+$ ]]; then
  echo "invalid 'last' value: '$last' (expected integer)" >&2
  exit 1
fi

next=$(( last + 1 ))

# Lock to avoid concurrent runs
lock="running"
if [[ -e "$lock" ]]; then
  echo "job running already (lock exists: $lock)" >&2
  exit 1
fi

# Always remove lock on exit, even on error/ctrl-c
cleanup() { rm -f "$lock"; }
trap cleanup EXIT INT TERM

gpu="$(hostname -s 2>/dev/null || uname -n)"
echo "host $gpu"

rm -f "requeue"
: > "run.out"
: > "error.out"
touch "$lock"

# call to program to do the work
set +e
assembly --run "$next" --nstep "$nstep" --tstep "$tstep" --gamma "$gamma" >"run.out" 2>"error.out"
py_rc=$?

# If python itself failed, treat as error (still allow CUDA parsing below)
if (( py_rc != 0 )); then
  echo "assembly.py exited with code $py_rc" >&2
fi

set -e

sleep 5

# Hard failure: illegal address -> don't requeue automatically
if grep -qE 'CUDA_ERROR_ILLEGAL_ADDRESS' "error.out"; then
  echo "problem with simulation (CUDA_ERROR_ILLEGAL_ADDRESS)" >&2
  exit 1
fi

# GPU/device failure -> drain node and request requeue (with delays)
if grep -qE 'CUDA_ERROR|No compatible CUDA device' "error.out"; then
  scontrol update "NodeName=$gpu" State=DRAIN Reason=gpudown || true
  echo "ok" > "requeue"
  echo gpu $gpu put in drain mode >&2
  sleep 5
  exit 0
fi

# Success path: only advance last if expected output exists
outxml="restart_${next}.xml"
if [[ -s "$outxml" ]]; then
  echo "$next" > "last"

  # Determine maxrun: prefer run-local, then global, else default
  maxrun=10
  if [[ -r "maxrun" ]]; then
    maxrun="$(< "maxrun")"
  elif [[ -r "maxrun" ]]; then
    maxrun="$(< "maxrun")"
  fi
  if ! [[ "$maxrun" =~ ^[0-9]+$ ]]; then
    echo "invalid maxrun value: '$maxrun' (expected integer); using 10" >&2
    maxrun=10
  fi

  if (( next < maxrun )); then
    echo "ok" > "requeue"
    sleep 5
  fi
else
  echo "expected output missing or empty: '$outxml'" >&2
  exit 1
fi

sleep 5

