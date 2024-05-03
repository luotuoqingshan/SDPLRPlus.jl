#!/bin/bash
declare -a arr=("ca-AstroPh" "ca-CondMat" "ca-GrQc" "ca-HepPh" "ca-HepTh" "email-Enron" "deezer_europe" "musae_facebook")

problem="MinimumBisection"

for i in {1..67}; do
    timeout 3600 /p/mnt/software/julia-1.10/bin/julia --project exp_CSDP.jl --graph "G$i" --problem "$problem"
    sleep 5
done

for i in "${arr[@]}"
do 
    timeout 3600 /p/mnt/software/julia-1.10/bin/julia --project exp_CSDP.jl --graph "$i" --problem "$problem"
    sleep 5 
done
