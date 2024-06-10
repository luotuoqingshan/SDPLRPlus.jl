#!/bin/bash

# Base URL for DIMACS10 matrices
base_url="https://www.cise.ufl.edu/research/sparse/matrices/DIMACS10"

# Get list of all .mat file links
wget -q -O- $base_url | grep -Po 'href="\K[^"]*\.mat' | sort -u > mat_files.txt

# Download each .mat file
while read -r line; do
    wget "${base_url}/${line}"
done < mat_files.txt

