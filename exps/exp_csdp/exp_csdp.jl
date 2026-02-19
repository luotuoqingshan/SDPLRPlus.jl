using CSDP
using JSON, MAT
using Random
using LinearAlgebra, Arpack
using SDPLRPlus: SymLowRankMatrix

function csdp_solver(
    C::TC,
    As::Vector{Any},
    bs::Vector{Tv},
    extra_f::Function,
    filename::String="csdp_result";
    seed::Int=0,
    filefolder::String=homedir()*"/SDPLRPlus.jl/output/",
    kwargs...,
) where {Tv<:AbstractFloat,TC<:AbstractMatrix{Tv}}
    Random.seed!(seed)
    dt = @elapsed begin
        csdp_C = BlockMatrix(Matrix(C))
        csdp_As = ConstraintMatrix[]
        for (i, A) in enumerate(As)
            push!(csdp_As, ConstraintMatrix(i, A))
        end
        params = default_params()
        for (k, v) in kwargs
            if k in keys(params)
                params[k] = v
            end
        end
        constraints = [A.csdp for A in csdp_As]
        X, y, Z = initsoln(csdp_C, bs, constraints)
        status, pobj, dobj = CSDP.parametrized_sdp(
            csdp_C, bs, csdp_As, X, y, Z, params
        )
    end
    extra_fval = extra_f(C, Matrix(X))
    res = Dict(
        "pobj" => pobj,
        "dobj" => dobj,
        "status" => status,
        "time" => dt,
        "seed" => seed,
        "X" => Matrix(X),
        "y" => Vector(y),
        "Z" => Matrix(Z),
        "params" => params,
        "extra_fval" => extra_fval,
    )
    @show pobj, dobj, status, dt, seed, extra_fval
    # short result 
    short_res = Dict(
        "pobj" => pobj,
        "dobj" => dobj,
        "status" => status,
        "time" => dt,
        "seed" => seed,
        "params" => params,
        "extra_fval" => extra_fval,
    )
    mkpath(filefolder)
    open(
        filefolder*filename*"-short-seed-$seed-tol-$(params[:axtol]).json", "w"
    ) do f
        json_str = JSON.json(short_res; pretty=4, allownan=true)
        print(f, json_str)
    end
end

function default_params()
    return Dict(
        :axtol => 1.0e-2,
        :atytol => 1.0e-2,
        :objtol => 1.0e-2,
        :pinftol => 1.0e8,
        :dinftol => 1.0e8,
        :maxiter => 100,
        :minstepfrac => 0.90,
        :maxstepfrac => 0.97,
        :minstepp => 1.0e-8,
        :minstepd => 1.0e-8,
        :usexzgap => 1,
        :tweakgap => 0,
        :affine => 0,
        :printlevel => 1,
        :perturbobj => 1,
        :fastmode => 0,
    )
end

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "--graph"
    arg_type = String
    default = "G1"
    help = "Name of the graph"
    "--problem"
    arg_type = String
    default = "MinimumBisection"
    help = "Name of the problem(MaxCut, MinimumBisection, LovaszTheta)"
    "--mu"
    arg_type = Float64
    default = 0.01
    help = "Parameter for mu-conductance"
    "--axtol"
    arg_type = Float64
    default = 1e-2
    help = "Tolerance for primal feasibility"
    "--atytol"
    arg_type = Float64
    default = 1e-2
    help = "Tolerance for dual feasibility"
    "--objtol"
    arg_type = Float64
    default = 1e-2
    help = "Tolerance for relative duality gap"
    "--seed"
    arg_type = Int
    default = 0
    help = "Random seed"
end

function read_graph(name, problem)
    if problem in ["LovaszTheta"]
        problem = "MaxCut"
    end
    return matread(
        "/p/mnt/data/yufan/datasets/graphs/"*problem*"/"*name*".mat"
    )["A"]
end

function eval_cut(L, x)
    # make sure x is a vector of +1 and -1
    return 0.25 * x' * L * x
end

function maxcut_rounding(C, Rt)
    R = Rt'
    r = size(R, 2)
    # take 100 cuts
    best_cut = -Inf
    L = 4 * C
    for _ in 1:100
        z = sign.(R * randn(r))
        best_cut = max(best_cut, eval_cut(L, z))
    end
    return best_cut
end

function minimumbisection_rounding(C, Rt)
    R = Rt'
    r = size(R, 2)
    # take 100 cuts
    L = -4 * C
    best_cut = Inf
    for _ in 1:100
        z = R * randn(r)
        perm = sortperm(z)
        n = length(perm)
        part = zeros(n)
        part[perm] .= [i*2 <= n for i in 1:n]*2 .- 1
        @assert sum(part) == 0
        best_cut = min(best_cut, eval_cut(L, part))
    end
    return best_cut
end

identity(C, Rt) = Inf

args = parse_args(s)
Random.seed!(args["seed"])
BLAS.set_num_threads(1)

include("../problems.jl")

if args["problem"] == "MaxCut"
    A = read_graph(args["graph"], args["problem"])
    C, As, bs = maxcut(A)
    extra_f = maxcut_rounding
elseif args["problem"] == "MinimumBisection"
    A = read_graph(args["graph"], args["problem"])
    C, As, bs = minimum_bisection(A)
    extra_f = minimumbisection_rounding
elseif args["problem"] == "LovaszTheta"
    A = read_graph(args["graph"], args["problem"])
    C, As, bs = lovasz_theta(A)
    extra_f = identity
elseif args["problem"] == "CutNorm"
    A = read_graph(args["graph"], args["problem"])
    C, As, bs = cutnorm(A)
    extra_f = identity
elseif args["problem"] == "MuConductance"
    A = read_graph(args["graph"], args["problem"])
    C, As, bs = mu_conductance(A, args["mu"])
    extra_f = identity
else
    error("Problem not supported")
end

problem = args["problem"]
graph = args["graph"]

# CSDP.jl is just a wrapper, no need to warm up
println("Solving ($problem) on graph ($graph) using CSDP.")
# notice that the primal problem of CSDP is a maximization problem
csdp_solver(
    -C,
    As,
    bs,
    extra_f,
    "csdp";
    filefolder=homedir() *
               "/SDPLRPlus.jl/exps/output/" *
               args["problem"] *
               "/$graph/",
    :axtol => args["axtol"],
    :atytol => args["atytol"],
    :objtol => args["objtol"],
)
