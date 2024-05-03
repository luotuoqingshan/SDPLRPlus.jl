using CSDP
using JSON
using Random
using Arpack

function csdp_solver(
    C::TC,
    As::Vector{Any},
    bs::Vector{Tv},
    extra_f::Function,
    filename::String="csdp_result";
    seed::Int=0,
    filefolder::String=homedir()*"/SDPLR.jl/output/",
    kwargs...
) where {Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    Random.seed!(seed)
    dt = @elapsed begin
        csdp_C =  BlockMatrix(Matrix(C))
        csdp_As = ConstraintMatrix[]
        for (i, A) in enumerate(As)
            push!(csdp_As, ConstraintMatrix(i, A))
        end
        params = default_params()
        for(k, v) in kwargs
            if k in keys(params)
                params[k] = v
            end
        end
        constraints = [A.csdp for A in csdp_As]
        X, y, Z = initsoln(csdp_C, bs, constraints)
        status, pobj, dobj = CSDP.parametrized_sdp(csdp_C, bs, csdp_As, X, y, Z, params)
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
    #open(filefolder*filename*"-seed-$seed.json", "w") do f
    #    JSON.print(f, res, 4)
    #end
    open(filefolder*filename*"-short-seed-$seed-tol-$(params[:axtol]).json", "w") do f
        JSON.print(f, short_res, 4)
    end
end


function default_params()
    return Dict(
        :axtol       =>   1.0e-2, 
        :atytol      =>   1.0e-2, 
        :objtol      =>   1.0e-2, 
        :pinftol     =>   1.0e8, 
        :dinftol     =>   1.0e8, 
        :maxiter     =>   100, 
        :minstepfrac =>   0.90, 
        :maxstepfrac =>   0.97, 
        :minstepp    =>   1.0e-8, 
        :minstepd    =>   1.0e-8, 
        :usexzgap    =>   1, 
        :tweakgap    =>   0, 
        :affine      =>   0, 
        :printlevel  =>   1, 
        :perturbobj  =>   1, 
        :fastmode    =>   0, 
    )
end


include("header.jl")
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
end

function maxcut_rounding(C, X)
    Z = svds(X, nsv=10)[1]
    # take 100 cuts
    best_cut = 0
    for i = 1:100
        z = sign.(Z.U * randn(10))
        best_cut = max(best_cut, z' * C * z)
    end
    return best_cut
end

args = parse_args(s)
BLAS.set_num_threads(1)

if args["problem"] == "MaxCut"
    A = read_graph(args["graph"])["A_abs"]
    C, As, bs = maxcut(A)
    extra_f = maxcut_rounding
elseif args["problem"] == "MinimumBisection"
    A = read_graph(args["graph"])["A_dummy_abs"]
    C, As, bs = minimum_bisection(A)
elseif args["problem"] == "LovaszTheta"
    A = read_graph(args["graph"])["A_abs"]
    C, As, bs = lovasz_theta(A)
elseif args["problem"] == "Cutnorm"
    A = read_graph(args["graph"])["A"]
    C, As, bs = cutnorm(A)
else
    error("Problem not supported")
end

problem = args["problem"]
graph = args["graph"]

# CSDP.jl is just a wrapper, no need to warm up
println("Solving ($problem) on graph ($graph) using CSDP.")
# notice that the primal problem of CSDP is a maximization problem
csdp_solver(-C, As, bs, extra_f, "csdp"; 
    filefolder=homedir()*"/SDPLR.jl/output/"*
    args["problem"]*"/$graph/csdp/",
    :axtol => args["axtol"], 
    :atytol => args["atytol"], 
    :objtol => args["objtol"])
