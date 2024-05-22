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


function batch_eval_cutnorm(name, filename::String; kwargs...)
    # warmup
    @info "Running Cut Norm on $name."
    A = matread(homedir()*"/datasets/graphs/CutNorm/"*name*".mat")["A"] 
    C, As, bs = cutnorm(A)
    n = size(C, 1)
    res = sdplr(C, As, bs, 10; prior_trace_bound=Float64(n), dataset=name, kwargs...)
    @show name, res["totaltime"], res["primaltime"], res["rel_duality_bound"]

    short_res_keys = ["grad_norm", "primal_vio", "obj", "rel_duality_bound", 
        "totaltime", "dual_lanczos_time", "dual_GenericArpack_time", 
        "primaltime", "iter", "majoriter", "ptol", "objtol", "fprec", 
        "rankupd_tol", "r",]
    short_res = Dict(k => res[k] for k in short_res_keys)

    output_folder = homedir()*"/SDPLR.jl/output/CutNorm/"
    mkpath(output_folder*"$name/SDPLR/")
    open(output_folder*"$name/SDPLR/"*filename*".json", "w") do f
        res["git commit"] = strip(read(`git rev-parse --short HEAD`, String))
        JSON.print(f, short_res, 4)
    end
end

# warmup
batch_eval_cutnorm("G1", "SDPLR-warmup"; maxtime=36000.0, objtol=args["objtol"], ptol=args["ptol"])
batch_eval_cutnorm("$(args["graph"])", "SDPLR-seed-$seed-tol-$tol"; maxtime=36000.0, objtol=args["objtol"], ptol=args["ptol"])
