#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <bias> <nstep>" >&2
  exit 2
}

bias="${1:-}"; nstep="${2:-}"
[[ -n "$bias" && -n "$nstep" ]] || usage

tstep=0.03   
gamma=1.0

tdir="run_${bias}"

if [[ ! -d "$tdir" ]]; then
  echo "directory '$tdir' does not exist" >&2
  exit 1
fi

# Read last (default 0)
last=0
if [[ -r "$tdir/last" ]]; then
  last="$(< "$tdir/last")"
fi
if ! [[ "$last" =~ ^[0-9]+$ ]]; then
  echo "invalid '$tdir/last' value: '$last' (expected integer)" >&2
  exit 1
fi

next=$(( last + 1 ))

# Lock to avoid concurrent runs
lock="$tdir/running"
if [[ -e "$lock" ]]; then
  echo "job running already (lock exists: $lock)" >&2
  exit 1
fi

# Always remove lock on exit, even on error/ctrl-c
cleanup() { rm -f "$lock"; }
trap cleanup EXIT INT TERM

gpu="$(hostname -s 2>/dev/null || uname -n)"
echo "host $gpu"

rm -f "$tdir/requeue"
: > "$tdir/run.out"
: > "$tdir/error.out"
touch "$lock"

#script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# call to program to do the work
set +e
distumbrella_cocomo --bias "$bias" --run "$next" --nstep "$nstep" --tstep "$tstep" --gamma "$gamma" >"$tdir/run.out" 2>"$tdir/error.out"

py_rc=$?

# If python itself failed, treat as error (still allow CUDA parsing below)
if (( py_rc != 0 )); then
  echo "distumbrella_cocomo exited with code $py_rc" >&2
fi
set -e

sleep 5

# Hard failure: illegal address -> don't requeue automatically
if grep -qE 'CUDA_ERROR_ILLEGAL_ADDRESS' "$tdir/error.out"; then
  echo "problem with simulation (CUDA_ERROR_ILLEGAL_ADDRESS)" >&2
  exit 1
fi

# GPU/device failure -> drain node and request requeue (with delays)
if grep -qE 'CUDA_ERROR|No compatible CUDA device' "$tdir/error.out"; then
  scontrol update "NodeName=$gpu" State=DRAIN Reason=gpudown || true
  echo "ok" > "$tdir/requeue"
  sleep 5 
  exit 0
fi

# Success path: only advance last if expected output exists
outxml="$tdir/biasprod_${next}.xml"
if [[ -s "$outxml" ]]; then
  echo "$next" > "$tdir/last"

  # Determine maxrun: prefer run-local, then global, else default
  maxrun=10
  if [[ -r "$tdir/maxrun" ]]; then
    maxrun="$(< "$tdir/maxrun")"
  elif [[ -r "maxrun" ]]; then
    maxrun="$(< "maxrun")"
  fi
  if ! [[ "$maxrun" =~ ^[0-9]+$ ]]; then
    echo "invalid maxrun value: '$maxrun' (expected integer); using 10" >&2
    maxrun=10
  fi

  if (( next < maxrun )); then
    echo "ok" > "$tdir/requeue"
    sleep 5
  fi
else
  echo "expected output missing or empty: '$outxml'" >&2
  exit 1
fi

sleep 5

