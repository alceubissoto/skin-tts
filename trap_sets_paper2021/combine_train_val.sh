#!/bin/bash

# Declare a string array with type
declare -a bias_factors=("0" "0.3" "0.5" "0.7" "0.9" "1")
declare -a runs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

# Read the array values with space
for b in "${bias_factors[@]}"; do
    for r in "${runs[@]}"; do
        head -1 train_bias_${b}_${r}.csv > trainval_bias_${b}_${r}.csv; tail -n +2 -q train_bias_${b}_${r}.csv val_bias_${b}_${r}.csv >> trainval_bias_${b}_${r}.csv
    done
done
