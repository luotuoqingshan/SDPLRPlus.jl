using Distributed
@everywhere include("header.jl")

Random.seed!(0)

@everywhere function f(i, filename::String)
    # warmup
    A = load_gset("G1") 
    C, As, bs = maxcut(A)
    n = size(A, 1)
    r = barvinok_pataki(n, n)
    res = sdplr(C, As, bs, r)

    # real start 
    A = load_gset("G$i")
    C, As, bs = maxcut(A)
    n = size(A, 1)
    r = barvinok_pataki(n, n)  

    res = sdplr(C, As, bs, r)

    output_folder = homedir()*"/SDPLR-jl/output/MaxCut/"
    matwrite(output_folder*"G$i/"*filename*".mat", res)
end

function benchmark_gset(gset_ids, filename::String)
    pmap(i -> f(i, filename), gset_ids)                
end

function print_gset_time(gset_ids, filename::String)
    for i in gset_ids
        output_folder = homedir()*"/SDPLR-jl/output/MaxCut/"
        res = matread(output_folder*"G$i/"*filename*".mat")
        @printf("G%d: primal time: %.3lf (s), dual time: %.3lf (s)\n",i, res["primaltime"], res["dualtime"])
    end
end

#f(1, "base")
#print_gset_time(1, "base")
#print_gset_time(48:67, "base")

benchmark_gset(48:67, "change_lbfgs")
print_gset_time(48:67, "base")
print_gset_time(48:67, "change_lbfgs")
#for i = 1:1 
#    A = load_gset("G$i")
#    C, As, bs = maxcut(A)
#    n = size(A, 1)
#    r = barvinok_pataki(n, n)  
#    size(A)
#    nnz(A)
#
#    res = sdplr(C, As, bs, r)
#
#    output_folder = homedir()*"/SDPLR-jl/output/MaxCut/"
#    mkdir(output_folder*"G$i/")
#    matwrite(output_folder*"G$i/SDPLR.mat", res)
#end
