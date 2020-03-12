#!/bin/bash

# Collect samples for the highway network at different inflows.
#
# TODO: description
#
# Usage:
#
#   sample_highway.sh [--samples SAMPLES]

# Default values of arguments
SAMPLES="50"

# Loop through arguments and process them
for arg in "$@"
do
  case $arg in
    -h)
    echo "usage: sample_highway.sh [--n_samples N_SAMPLES]"
    echo ""
    echo "arguments"
    echo "  -h, --help            show this help message and exit"
    echo "  --samples SAMPLES     the number of samples to collect for each inflow rate."
    echo "                        Defaults to 50."
    shift
    ;;
    --n_samples)
    SAMPLES="$2"
    shift # Remove argument name from processing
    shift # Remove argument value from processing
    ;;
  esac
done
