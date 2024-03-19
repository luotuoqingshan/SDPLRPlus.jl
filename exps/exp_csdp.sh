#!/bin/bash

for n in {1..67};
do 
    timeout 3600 /p/mnt/software/julia-1.10/bin/julia --project exp_CSDP.jl --graph "G${n}" --problem "MaxCut"
    sleep 60
done