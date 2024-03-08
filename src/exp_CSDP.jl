using CSDP
using JSON

function csdp_solver(
    C::TC,
    As::Vector{Any},
    bs::Vector{Tv},
    filename::String="csdp_result";
    filefolder::String=homedir()*"/SDPLR-jl/output/",
    kwargs...
) where {Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
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
    res = Dict(
        "pobj" => pobj,
        "dobj" => dobj,
        "status" => status,
        "time" => dt,
        "X" => Matrix(X),
        "y" => Vector(y),
        "Z" => Matrix(Z),
        "params" => params,
    )
    open(filefolder*filename*".json", "w") do f
        JSON.print(f, res, 4)
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

args = parse_args(s)

A = read_graph(args["graph"])

if args["problem"] == "MaxCut"
    C, As, bs = maxcut(A)
elseif args["problem"] == "MinimumBisection"
    C, As, bs = minimum_bisection(A)
elseif args["problem"] == "LovaszTheta"
    C, As, bs = lovasz_theta(A)
else
    error("Problem not supported")
end

problem = args["problem"]
graph = args["graph"]
println("Solving ($problem) on graph ($graph) using CSDP.")
# notice that the primal problem of CSDP is a maximization problem
csdp_solver(-C, As, bs, args["problem"]*"/"*args["graph"]*"/csdp"; 
    :axtol => args["axtol"], :atytol => args["atytol"], :objtol => args["objtol"])