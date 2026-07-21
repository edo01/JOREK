#!/bin/bash

#
# Purpose: Modify or print physics model in Makefile.inc and parameters in
# mod_settings.h + mod_settings.f90 (strictly synchronized) and model-specific settings.
#

function usage() {
  echo ""
  echo "Usage: $(basename $0) [key=value ...]"
  echo "       $(basename $0) -p <key>"
  echo ""
  echo "Examples:"
  echo "  $(basename $0) model=302 n_tor=7 n_plane=8"
  echo "  $(basename $0) -p n_tor"
  echo ""
}

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  usage
  exit 0
elif [ ! -f "Makefile.inc" ]; then
  echo "ERROR: Could not find Makefile.inc. Are you in the JOREK trunk?" >&2
  exit 1
fi

# -------------------------
# helpers
# -------------------------

key() { echo "$1" | sed 's/=.*$//'; }
val() { echo "$1" | sed 's/^.*=//'; }

# -------------------------
# model handling
# -------------------------

function getmodel() {
  grep "MODEL *= *model[0-9]*" Makefile.inc | \
    sed -E "s/^ *MODEL *= *(model[0-9]+).*$/\1/"
}

function setmodel() {
  model=$1

  if [ -e "models/model$model" ]; then
    model="model$model"
  elif [ ! ${#model} -eq 8 ] || [[ ! ${model:5:3} =~ ^[0-9]+$ ]]; then
    echo "ERROR: Illegal model '$model'" >&2
    exit 1
  fi

  sed -i -E "s/(^ *MODEL *= *)[^ ]+/\1$model/" Makefile.inc
  make cleanall
}

# -------------------------
# parameter files
# -------------------------

model=$(getmodel)

paramfile_h="models/mod_settings.h"
paramfile_f90="models/mod_settings.f90"
paramfile_model_f90="models/$model/mod_model_settings.f90"

paramfiles="$paramfile_h $paramfile_f90 $paramfile_model_f90"

for f in $paramfiles; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Missing file $f" >&2
    exit 1
  fi
done

# -------------------------
# parameter setter (SYNC MODE)
# -------------------------

setparam() {
  key=$1
  val=$2

  # must exist in BOTH base files (strict mirror rule)
  if ! grep -qE "^[[:space:]]*#define[[:space:]]+$key[[:space:]]+" "$paramfile_h"; then
    echo "ERROR: $key not found in mod_settings.h" >&2
    exit 1
  fi

  if ! grep -qE "^[[:space:]]*integer[[:space:]]*,[[:space:]]*parameter[[:space:]]*::[[:space:]]*$key[[:space:]]*=" "$paramfile_f90"; then
    echo "ERROR: $key not found in mod_settings.f90" >&2
    exit 1
  fi

  # update .h
  sed -i -E \
    "s/^([[:space:]]*#define[[:space:]]+$key[[:space:]]+)[^[:space:]]+/\1$val/" \
    "$paramfile_h"

  # update .f90
  sed -i -E \
    "s/(^[[:space:]]*integer[[:space:]]*,[[:space:]]*parameter[[:space:]]*::[[:space:]]*$key[[:space:]]*=[[:space:]]*)[^![:space:]]+/\1$val/" \
    "$paramfile_f90"

  echo "Updated $key = $val (synced .h + .f90)"
}

# -------------------------
# parameter getter
# -------------------------

getparam() {
  key=$1

  for f in $paramfiles; do
    if [[ "$f" == *.h ]]; then
      awk -v k="$key" '$1=="#define" && $2==k {print $3; exit}' "$f"
    else
      grep -E "^[[:space:]]*integer[[:space:]]*,[[:space:]]*parameter[[:space:]]*::[[:space:]]*$key[[:space:]]*=" "$f" | \
        sed -E "s/^.*=[[:space:]]*([^![:space:]]+).*/\1/"
    fi
  done | head -n 1
}

# -------------------------
# print info
# -------------------------

whichparams() {
  grep '#SETTINGS#' "$1" | sed 's/^.*#//'
}

print_info() {
  echo ""
  echo "=============================="
  echo "  $(getmodel)"
  echo "------------------------------"

  for p in $(whichparams "$paramfile_h"); do
    printf "  %-16s = %s\n" "$p" "$(getparam "$p")"
  done

  echo "=============================="
  echo ""
}

# -------------------------
# main logic
# -------------------------

# -p mode
if [ "$1" == "-p" ]; then
  if [ "$2" == "model" ]; then
    echo "$(getmodel | sed 's/model//')"
  else
    v=$(getparam "$2")
    if [ -z "$v" ]; then
      echo "Parameter not found: $2"
      exit 1
    fi
    echo "$v"
  fi
  exit 0
fi

# model first if provided
for arg in "$@"; do
  if [ "$(key "$arg")" == "model" ]; then
    setmodel "$(val "$arg")"
  fi
done

model=$(getmodel)
paramfile_model_f90="models/$model/mod_model_settings.f90"
paramfiles="$paramfile_h $paramfile_f90 $paramfile_model_f90"

# apply parameters
for arg in "$@"; do
  k=$(key "$arg")
  v=$(val "$arg")

  if [ "$k" != "model" ]; then
    setparam "$k" "$v"
  fi
done

print_info
echo "('$(basename $0) -h' for help)"