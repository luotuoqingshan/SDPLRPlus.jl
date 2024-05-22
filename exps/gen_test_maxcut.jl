graphs = ["G$i" for i = 1:9] 
seed = 0

open(homedir()*"/SDPLR.jl/exps/test_maxcut.txt", "w") do io
    for graph in graphs
        println(io, "ulimit -d $((16 * 1024 * 1024)); "*
        "cd ~/SDPLR.jl/exps; "*
        "/p/mnt/software/julia-1.10/bin/julia "*
        "--project test_maxcut.jl --seed $seed --graph \"$graph\"")
    end
end