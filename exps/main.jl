#using Distributed
using Logging
using JSON
include("header.jl")
Random.seed!(0)

logger = SimpleLogger(stdout, Logging.Info)
old_logger = global_logger(logger)

function batch_eval_maxcut(i, filename::String)
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


function batch_eval_minimum_bisection(i, filename::String)
    @info "Running minimum bisection on G$i."
    A = read_graph("G$i")
    C, As, bs = minimum_bisection(A)
    r = barvinok_pataki(size(A, 1), length(As))
    res = sdplr(C, As, bs, 5)

    output_folder = homedir()*"/SDPLR-jl/output/MinimumBisection/"
    mkpath(output_folder*"G$i/")
    open(output_folder*"G$i/"*filename*".json", "w") do f
        res["git commit"] = strip(read(`git rev-parse --short HEAD`, String))
        JSON.print(f, res, 4)
    end
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

#A = read_graph("G1")
#C, As, bs = minimum_bisection(A)
#n = size(A, 1)
#r = barvinok_pataki(n, length(As))  
#@timed res = sdplr(C, As, bs, 5)

batch_eval_minimum_bisection(1, "Mar-5-2024")
batch_eval_minimum_bisection(50, "Mar-5-2024")


#for i = 51:67
#    batch_eval_minimum_bisection(i, "Mar-5-2024") 
#end

