using Distributed
@everywhere include("header.jl")

Random.seed!(0)

@everywhere function batch_eval_maxcut(i, filename::String)
    # warmup
    A = read_graph("G1") 
    C, As, bs = maxcut(A)
    n = size(A, 1)
    r = barvinok_pataki(n, length(As))
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


@everywhere function batch_eval_minimum_bisection(i, filename::String)
    @info "Running minimum bisection on G$i."
    A = read_graph("G$i")
    C, As, bs = minimum_bisection(A)
    n = size(A, 1)
    r = barvinok_pataki(n, length(As))  

    res = sdplr(C, As, bs, r)

    output_folder = homedir()*"/SDPLR-jl/output/MinimumBisection/"
    mkpath(output_folder*"G$i/")
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

#warmup
batch_eval_minimum_bisection(1, "tol-1e-2")

# real start
for i = 1:67
    batch_eval_minimum_bisection(i, "tol-1e-2")
end

A = read_graph("G57")
C, As, bs = lovasz_theta(A)
n = size(A, 1)
r = barvinok_pataki(n, length(As))  

@timed res = sdplr(C, As, bs, 5)