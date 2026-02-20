#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

tag="${1:-run}"

for n in set*; do
  # Only process directories
  [[ -d "$n" ]] || continue

  nset="${n#set}"  # everything after "run_"

  (
    cd "$n" || exit 0

    cur_base=$(basename -- "$PWD")               # e.g. set1
    parent=$(basename -- "$(dirname -- "$PWD")") # e.g. ht
    setnum=${cur_base#set}                       # e.g. 1
    settag="${parent}_${setnum}"

    logs=( "${tag}"_*.log )
    [[ ${#logs[@]} -gt 0 ]] || exit 0

    newest_log="$(ls -1t "${tag}"_*.log 2>/dev/null | head -n 1 || true)"
    [[ -n "$newest_log" ]] || exit 0

    # Date string from stat (portable-ish: GNU stat uses -c; macOS uses -f)
    if date_str="$(stat -c '%y' "$newest_log" 2>/dev/null | awk '{print $1, $2}' )"; then
      : # ok
    else
      date_str="$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$newest_log" 2>/dev/null || echo "-")"
    fi

    host="-"
    if [[ -r "slurm.prod.out" ]]; then
      host="$(awk '/host/ {print $2; exit}' slurm.prod.out || true)"
      [[ -n "$host" ]] || host="-"
    else
      slurm_out=( slurm*.out )
      if [[ ${#slurm_out[@]} -gt 0 && -r "${slurm_out[0]}" ]]; then
        host="$(awk '/host/ {print $2; exit}' "${slurm_out[0]}" || true)"
        [[ -n "$host" ]] || host="-"
      fi
    fi

    last=0
    if [[ -r "last" ]]; then
      last="$(< last)"
    fi

    qcount="$(squeue -h 2>/dev/null | grep -F -c "$settag" || true)"
    lastline="$(tail -n 1 "$newest_log" 2>/dev/null | tr -d '\r' || echo "-")"

    printf "%s %s %s %s %s %s %s\n" \
      "$settag" "$qcount" "$last" "$date_str" "$newest_log" "$host" "$lastline"
  )
done

