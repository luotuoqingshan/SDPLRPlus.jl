using Logging
using JSON
using Profile
using ArgParse
include("header.jl")

s = ArgParseSettings()

@add_arg_table s begin
    "--graph"
        arg_type = String
        default = "G1" 
        help = "Name of the graph"
    "--ptol"
        arg_type = Float64
        default = 1e-2
        help = "Tolerance for primal feasibility"
    "--objtol" 
        arg_type = Float64
        default = 1e-2
        help = "Tolerance for relative duality gap"
    "--seed"
        arg_type = Int
        default = 0
        help = "Random seed"
end

args = parse_args(s)
seed = args["seed"]
tol = args["ptol"]
Random.seed!(seed)
BLAS.set_num_threads(1)

logger = SimpleLogger(stdout, Logging.Info)
old_logger = global_logger(logger)

function eval_cut(L, x) 
    # make sure x is a vector of +1 and -1
    return 0.25 * x' * L * x
end

function minimumbisection_rounding(L, Rt)
    R = Rt'
    r = size(R, 2)
    # take 100 cuts
    best_cut = Inf
    for _ = 1:100
        z = R * randn(r)
        perm = sortperm(z)
        n = length(perm)
        part = zeros(n)
        part[perm] .= [i*2 <= n for i=1:n]*2 .- 1
        @assert sum(part) == 0
        best_cut = min(best_cut, eval_cut(L, part))
    end
    return best_cut
end

function batch_eval_minimumbisection(name, filename::String; kwargs...)
    # warmup
    @info "Running minimum bisection on $name."
    A = matread(homedir()*"/datasets/graphs/MinimumBisection/"*name*".mat")["A"] 
    C, As, bs = minimum_bisection(A)
    n = size(A, 1)
    res = sdplr(C, As, bs, 10; prior_trace_bound=Float64(n), dataset=name, kwargs...)
    L = Diagonal(sum(A, dims=1)[:]) - A
    best_cut = minimumbisection_rounding(L, res["Rt"])
    res["best_cut"] = best_cut
    @show name, res["totaltime"], res["primaltime"], res["rel_duality_bound"], res["best_cut"]

    short_res_keys = ["grad_norm", "primal_vio", "obj", "rel_duality_bound", 
        "totaltime", "dual_lanczos_time", "dual_GenericArpack_time", 
        "primaltime", "iter", "majoriter", "ptol", "objtol", "fprec", 
        "best_cut", "rankupd_tol", "r",]
    short_res = Dict(k => res[k] for k in short_res_keys)

    output_folder = homedir()*"/SDPLR.jl/output/MinimumBisection/"
    mkpath(output_folder*"$name/SDPLR/")
    open(output_folder*"$name/SDPLR/"*filename*".json", "w") do f
        res["git commit"] = strip(read(`git rev-parse --short HEAD`, String))
        JSON.print(f, short_res, 4)
    end
end

# warmup
batch_eval_minimumbisection("G1", "SDPLR-warmup"; maxtime=36000.0, objtol=args["objtol"], ptol=args["ptol"])
batch_eval_minimumbisection("$(args["graph"])", "SDPLR-seed-$seed-tol-$tol";
                            maxtime=36000.0, objtol=args["objtol"], ptol=args["ptol"])

