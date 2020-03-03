#!/bin/bash

# Collect samples for the merge network at different inflows.
#
# TODO: description
#
# Usage:
#
#   sample_merge.sh [--lanes LANES] [--samples SAMPLES]

# Default values of arguments
LANES="1"
SAMPLES="50"

# Loop through arguments and process them
for arg in "$@"
do
  case $arg in
    -h)
    echo "usage: sample_merge.sh [--lanes LANES] [--samples SAMPLES]"
    echo ""
    echo "arguments"
    echo "  -h, --help            show this help message and exit"
    echo "  --lanes LANES         the number of lanes in the ring road. Defaults to 1."
    echo "  --samples SAMPLES     the number of samples to collect for each network density."
    echo "                        Defaults to 50."
    shift
    ;;
    --lanes)
    LANES="$2"
    shift # Remove argument name from processing
    shift # Remove argument value from processing
    ;;
    --samples)
    SAMPLES="$2"
    shift # Remove argument name from processing
    shift # Remove argument value from processing
    ;;
  esac
done

# Declare the list of # vehicles to loop through
declare -a StringArray=("1000" "1100" "1200" "1300" "1400"
                        "1500" "1600" "1700" "1800" "1900" "2000")

# Read the array values with space
for val in "${StringArray[@]}"; do
  # Scale inflow rate by number of lanes
  INFLOW=$((val * LANES))

  # Updates the inflow rate in the script
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  sed -i -e "s/.*FLOW_RATE = .*/FLOW_RATE = $INFLOW/" "$DIR"/../../mbrl_traffic/envs/params/merge.py

  echo ""
  echo "Starting inflow rate: $INFLOW"
  echo "--------------------------"

  # Run the simulations for the given number of samples
  for _ in $(seq 1 "$SAMPLES"); do
    python simulate.py "merge" --no_render --gen_emission
  done
done
