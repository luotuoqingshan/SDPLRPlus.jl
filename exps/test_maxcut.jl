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
    "--rank"
        arg_type = Int
        default = 10
        help = "Initial rank"
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

function maxcut_rounding(L, Rt)
    R = Rt'
    r = size(R, 2)
    # take 100 cuts
    best_cut = -Inf
    for _ = 1:100
        z = sign.(R * randn(r))
        best_cut = max(best_cut, eval_cut(L, z))
    end
    return best_cut
end

function batch_eval_maxcut(name, filename::String, r=10; kwargs...)
    # warmup
    @info "Running max cut on $name."
    A = matread(homedir()*"/datasets/graphs/MaxCut/"*name*".mat")["A"] 
    C, As, bs = maxcut(A)
    n = size(A, 1)
    res = sdplr(C, As, bs, r; prior_trace_bound=Float64(n), dataset=name, kwargs...)
    L = Diagonal(sum(A, dims=1)[:]) - A
    best_cut = maxcut_rounding(L, res["Rt"])
    res["best_cut"] = best_cut
    @show name, res["totaltime"], res["primaltime"], res["rel_duality_bound"]

    short_res_keys = ["grad_norm", "primal_vio", "obj", "rel_duality_bound", 
        "totaltime", "dual_lanczos_time", "dual_GenericArpack_time", 
        "primaltime", "iter", "majoriter", "ptol", "objtol", "fprec", 
        "best_cut", "rankupd_tol", "r",]
    short_res = Dict(k => res[k] for k in short_res_keys)

    output_folder = homedir()*"/SDPLR.jl/output/MaxCut/"
    mkpath(output_folder*"$name/SDPLR/")
    open(output_folder*"$name/SDPLR/"*filename*".json", "w") do f
        res["git commit"] = strip(read(`git rev-parse --short HEAD`, String))
        JSON.print(f, short_res, 4)
    end
end

if VERSION >= v"1.7"
    @show LinearAlgebra.BLAS.get_config()
  end
  @show LinearAlgebra.BLAS.get_num_threads()
  @show get(()->"", ENV, "OPENBLAS_NUM_THREADS")
  @show get(()->"", ENV, "GOTO_NUM_THREADS")
  @show get(()->"", ENV, "OMP_NUM_THREADS")
  @show get(()->"", ENV, "MKL_NUM_THREADS")
  @show get(()->"", ENV, "OPENBLAS_BLOCK_FACTOR")
  @show Threads.nthreads()
# warmup
batch_eval_maxcut("G1", "SDPLR-warmup", args["rank"]; maxtime=36000.0, objtol=args["objtol"], ptol=args["ptol"])
batch_eval_maxcut("$(args["graph"])", "SDPLR-R-$(args["rank"])-seed-$seed-tol-$tol", args["rank"]; maxtime=36000.0, objtol=args["objtol"], ptol=args["ptol"])
