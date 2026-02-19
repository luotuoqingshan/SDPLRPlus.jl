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
    "--problem"
    arg_type = String
    default = "MaxCut"
    help = "Problem type"
    "--nthreads"
    arg_type = Int
    default = 1
    help = "Number of threads. For benchmarking, we recommend 1."
end

args = parse_args(s)

seed = args["seed"]
tol = args["ptol"]

Random.seed!(seed)
BLAS.set_num_threads(args["nthreads"])

"""
check env to ensure the number of threads is set to 1
when benchmarking.
"""
function nthread_check()
    if VERSION >= v"1.7"
        @show LinearAlgebra.BLAS.get_config()
    end
    @show LinearAlgebra.BLAS.get_num_threads()
    @show get(() -> "", ENV, "OPENBLAS_NUM_THREADS")
    @show get(() -> "", ENV, "GOTO_NUM_THREADS")
    @show get(() -> "", ENV, "OMP_NUM_THREADS")
    @show get(() -> "", ENV, "MKL_NUM_THREADS")
    @show get(() -> "", ENV, "OPENBLAS_BLOCK_FACTOR")
    @show Threads.nthreads()
end

# comment out if you don't want this info
nthread_check()

logger = SimpleLogger(stdout, Logging.Info)
old_logger = global_logger(logger)

function eval_cut(L, x)
    # make sure x is a vector of +1 and -1
    return 0.25 * x' * L * x
end

function maxcut_rounding(A, Rt)
    L = Diagonal(sum(A; dims=1)[:]) - A
    R = Rt'
    r = size(R, 2)
    # take 100 cuts
    best_cut = -Inf
    for _ in 1:100
        z = sign.(R * randn(r))
        best_cut = max(best_cut, eval_cut(L, z))
    end
    return best_cut
end

function minimumbisection_rounding(A, Rt)
    L = Diagonal(sum(A; dims=1)[:]) - A
    R = Rt'
    r = size(R, 2)
    # take 100 cuts
    best_cut = Inf
    for _ in 1:100
        z = R * randn(r)
        perm = sortperm(z)
        n = length(perm)
        part = zeros(n)
        part[perm] .= [i * 2 <= n for i in 1:n] * 2 .- 1
        @assert sum(part) == 0
        best_cut = min(best_cut, eval_cut(L, part))
    end
    return best_cut
end

dymmy_callback(A, Rt) = 0

function batch_eval(
    problem,
    graph,
    A,
    input::Function,
    callback::Function,
    filename::String,
    trace_bound,
    r=10;
    kwargs...,
)
    # warmup
    @info "Running max cut on $graph."
    C, As, bs = input(A)
    res = sdplr(
        C,
        As,
        bs,
        r;
        prior_trace_bound=Float64(trace_bound),
        dataset=graph,
        kwargs...,
    )
    callback_res = callback(A, res["Rt"])
    res["callback_res"] = callback_res
    @show graph, res["totaltime"], res["primaltime"], res["rel_duality_bound"]

    short_res_keys = [
        "grad_norm",
        "primal_vio",
        "obj",
        "rel_duality_bound",
        "totaltime",
        "dual_lanczos_time",
        "dual_GenericArpack_time",
        "primaltime",
        "iter",
        "majoriter",
        "ptol",
        "objtol",
        "fprec",
        "callback_res",
        "rankupd_tol",
        "r",
    ]
    short_res = Dict(k => res[k] for k in short_res_keys)

    output_folder = homedir() * "/SDPLRPlus.jl/exps/output/$problem/"
    mkpath(output_folder * "$graph/")
    open(output_folder * "$graph/" * filename * ".json", "w") do f
        #short_res["git commit"] = strip(read(`git rev-parse --short HEAD`, String))
        JSON.print(f, short_res, 4)
    end
end

problem = args["problem"]
graph = args["graph"]
ind = findfirst(
    x -> x == problem, ["MaxCut", "MinimumBisection", "LovaszTheta", "CutNorm"]
)
inputs = [maxcut, minimum_bisection, lovasz_theta, cutnorm]
callbacks = [
    maxcut_rounding, minimumbisection_rounding, dymmy_callback, dymmy_callback
]

A = matread("/p/mnt/data/yufan/datasets/graphs/$problem/G1.mat")["A"]
n = size(A, 1)
trace_bounds = [n, n, 1, n]

# warmup
# this is necessay for benchmarking julia code
batch_eval(
    problem,
    "G1",
    A,
    inputs[ind],
    callbacks[ind],
    "SDPLR-warmup",
    trace_bounds[ind],
    args["rank"];
    maxtime=36000.0,
    objtol=1.0,
    ptol=1.0,
)

A = matread("/p/mnt/data/yufan/datasets/graphs/$problem/$graph.mat")["A"]
n = size(A, 1)
trace_bounds = [n, n, 1, n]
# run the benchmark
batch_eval(
    problem,
    graph,
    A,
    inputs[ind],
    callbacks[ind],
    "SDPLR-R-$(args["rank"])-seed-$seed-tol-$tol",
    trace_bounds[ind],
    args["rank"];
    maxtime=36000.0,
    objtol=args["objtol"],
    ptol=args["ptol"],
)
