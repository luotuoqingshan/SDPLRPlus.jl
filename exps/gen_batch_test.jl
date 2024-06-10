graphs = ["G$i" for i = 1:9] 
seed = 0
problem = "MaxCut" # support ["MaxCut", "MinimumBisection", "LovaszTheta", "CutNorm]
tol = 0.01
initial_rank = 10

open(homedir()*"/SDPLRPlus.jl/exps/batch_test.txt", "w") do io
    for graph in graphs
        println(io, "ulimit -d $((16 * 1024 * 1024)); "*
        "cd ~/SDPLRPlus.jl/exps; "*
        "/p/mnt/software/julia-1.10/bin/julia "*
        "--project test.jl --seed $seed --graph \"$graph\" --problem \"$problem\" "*
        "--ptol $tol --objtol $tol --rank $initial_rank")
    end
end