#!/bin/bash

# Collect samples for the ring network at different densities.
#
# TODO
#
# Usage:
#
#   TODO

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
declare -a StringArray=("2000" "2100" "2200" "2300" "2400" "2500" "2600"
                        "2700" "2800" "2900" "3000")

# Read the array values with space
for val in "${StringArray[@]}"; do
  # Scale the inflow rate by the given number of lanes
  INFLOW_RATE=$((val * LANES))

  # Updates the number of vehicles in the script
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  sed -i -e "s/.*HIGHWAY_INFLOW_RATE = .*/HIGHWAY_INFLOW_RATE = $INFLOW_RATE/" "$DIR"/../../mbrl_traffic/envs/params/highway_single.py

  echo ""
  echo "Starting inflow rate: $INFLOW_RATE"
  echo "--------------------------------"

  # Run the simulations for the given number of samples
  for _ in $(seq 1 "$SAMPLES"); do
    python simulate.py "highway-single" --no_render --gen_emission
  done
done
