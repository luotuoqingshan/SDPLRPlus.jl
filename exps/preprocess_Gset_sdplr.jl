graphs = readlines(homedir()*"/datasets/graphs/GSet.txt")

include("header.jl")
for graph in graphs
    data = matread(homedir()*"/datasets/graphs/MaxCut/$graph.mat")
    A = data["A"]
    C, As, bs = maxcut(A)
    write_problem_sdpa("$(graph)_abs.sdpa", C, As, bs; filefolder=homedir()*"/SDPLR-1.03-beta-raw/data/")
end