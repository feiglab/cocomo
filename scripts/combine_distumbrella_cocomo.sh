#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

if [[ -n "${1:-}" && -d "$1" ]]; then
  cd -- "$1"
fi

nskip=2
setupdir="setup"
pdb="${setupdir}/dimer.protein.pdb"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Print size in bytes
fsize() {
  local f="$1"
  if stat -c%s "$f" >/dev/null 2>&1; then
    stat -c%s "$f"
  else
    stat -f%z "$f"
  fi
}

# Ensure all files have identical sizes
check_same_size() {
  local label="$1"
  shift
  local files=("$@")
  ((${#files[@]})) || return 0

  local ref f s
  ref="$(fsize "${files[0]}")"

  for f in "${files[@]}"; do
    s="$(fsize "$f")"
    if [[ "$s" != "$ref" ]]; then
      die "${label}: size mismatch: $f is ${s}B, expected ${ref}B"
    fi
  done
}

nprotein="$(grep -c '^ATOM' "$pdb")"
grep -E '( CA |TER|END)' "$pdb" > CA.pdb
nca="$(grep -c '^ATOM' CA.pdb)"

for rundir in run_*; do
  [[ -d "$rundir" ]] || continue
  (
    cd -- "$rundir"
    flist=(biasprod_{?,??}.dcd)
    ((${#flist[@]})) || exit 0

    check_same_size "Input DCDs in ${rundir}" "${flist[@]}"

    mdconv -out biasprod.skip.dcd -atoms "1:${nprotein}" -skip "$nskip" "${flist[@]}"
  )
done

mapfile -t skip_files < <(
  for f in run_*/biasprod.skip.dcd; do
    [[ -e "$f" ]] || continue

    dir="${f%/biasprod.skip.dcd}"
    name="${dir#run_}"

    if [[ "$name" =~ ^([0-9]+([.][0-9]+)?)(_[0-9]+([.][0-9]+)?)?$ ]]; then
      first="${BASH_REMATCH[1]}"

      if [[ -n "${BASH_REMATCH[3]:-}" ]]; then
        has_second=1
        second="${BASH_REMATCH[3]#_}"
      else
        has_second=0
        second=0
      fi

      printf '%s\t%s\t%s\t%s\n' \
        "$first" "$has_second" "$second" "$f"
    else
      die "Unexpected run directory name: $dir"
    fi
  done | sort -t $'\t' -k1,1g -k2,2n -k3,3g | cut -f4-
)

((${#skip_files[@]})) || die "No run_*/biasprod.skip.dcd files found."

check_same_size "biasprod.skip.dcd across run dirs" "${skip_files[@]}"

if (( nprotein > nca )); then
  mdconv -out biasprod.all.dcd "${skip_files[@]}"

  grep ATOM $pdb | awk '/ CA / {print NR}' > ca.atomlist
  mdconv -atomlist ca.atomlist -out ca.dcd biasprod.all.dcd
else
  mdconv -out ca.dcd "${skip_files[@]}"
fi

