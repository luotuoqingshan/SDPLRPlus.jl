using Distributed
@everywhere include("header.jl")

Random.seed!(0)

@everywhere function f(i, filename::String)
    # warmup
    A = read_graph("G1") 
    C, As, bs = maxcut(A)
    n = size(A, 1)
    r = barvinok_pataki(n, n)
    res = sdplr(C, As, bs, r)

    # real start 
    A = read_graph("G$i")
    C, As, bs = maxcut(A)
    n = size(A, 1)
    r = barvinok_pataki(n, length(As))  

    res = sdplr(C, As, bs, r)

    output_folder = homedir()*"/SDPLR-jl/output/MaxCut/"
    matwrite(output_folder*"G$i/"*filename*".mat", res)
end

function benchmark_gset(gset_ids, filename::String)
    pmap(i -> f(i, filename), gset_ids)                
end

function print_gset(
    gset_ids, 
    field::String, 
    program::String, 
    filename::String; 
    average::Bool=false,
)
    sum = 0.0
    for i in gset_ids
        output_folder = homedir()*"/SDPLR-jl/output/"*program*"/"
        res = matread(output_folder*"G$i/"*filename*".mat")
        @assert field in keys(res) "This field was not saved."
        sum += res[field]
        @info "G$i: $field: $(res[field])"
    end
    if average
        @info "Average $field: $(sum/length(gset_ids))"
    end
end

#f(66, "early_termination_1e-5_1e-4")
#for i = 1:67
#    f(i, "early_termination_1e-3_1e-3_1e-3")
#end
#print_gset(66:66, "primaltime", "MaxCut", "early_termination_1e-5_1e-4")
A = read_graph("G1")
C, As, bs = minimum_bisection(A);
#C, As, bs = lovasz_theta(A)
r = barvinok_pataki(size(C, 1), length(As))
res = sdplr(C, As, bs, r)

n = 20 
A = ones(n, n)
A[diagind(A)] .= 0
A = sparse(A)
dropzeros!(A)

write_graph(A, "K$n"; extension=".smat")
#
#C, As, bs = maxcut(A)
#write_problem_sdplr(C, As, bs, homedir()*"/SDPLR-jl/data/G1_lovasz_theta.sdplr")
