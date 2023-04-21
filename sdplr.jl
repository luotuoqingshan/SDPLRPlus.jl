include("structures.jl")
include("dataoper.jl")
include("lbfgs.jl")
include("myprint.jl")
include("linesearch.jl")

"""
Interface for SDPLR 

Problem formulation:
    min   Tr(C (YY^T))
    s.t.  Tr(As[i] (YY^T)) = bs[i]
            Y in R^{n x r}
"""
function sdplr(
    C::AbstractMatrix,
    As::Vector{AbstractMatrix},
    b::Vector,
    r::Int;
    ρ_f = 1e-5,
    ρ_c = 1e-1,
    σ_fac = 2.0,
    timelimit = 3600.0,
    printlevel = 1,
    numblbfgsvec = 3, 
    σ_strategy = 1,
    λ_updatect = 1,
    maxmajiter = 10^5,
    maxiter = 10^7,
)
    m = length(As)
    @assert (typeof(C) <: SparseMatrixCSC 
         || typeof(C) <: Diagonal
         || typeof(C) <: LowRankMatrix) "Wrong matrix type of cost matrix."
    A_sp = SparseMatrixCSC[]
    A_d = Diagonal[]
    A_lr = LowRankMatrix[]
    for i = 1:m
        if typeof(As[i]) <: SparseMatrixCSC
            push!(A_sp, As[i])
        elseif typeof(As[i]) <: Diagonal
            push!(A_d, As[i])
        elseif typeof(As[i]) <: LowRankMatrix
            push!(A_lr, As[i])
        else
            error("Wrong matrix type of constraint matrix.")
        end
    end
    pdata = ProblemData(m, length(A_sp), length(A_d), length(A_lr),
                        A_sp, A_d, A_lr, C, bs)
    n = size(C, 1)
    R_0 = 2 .* rand(n, r) .- 1
    algdata = AlgorithmData(
        randn(m),         #λ
        1.0 / n,          #σ
        0,                #obj, will be initialized later
        zeros(m),         #vio(violation + obj), will be initialized later
        R_0,              #R
        zeros(size(R_0)), #G, will be initialized later
        time(),           #starttime
    )
    config = Config(ρ_f, ρ_c, σ_fac, timelimit, printlevel, 
                    numblbfgsvec, σ_strategy, λ_updatect, maxmajiter, maxiter)
    res = _sdplr(pdata, algdata, config)
    return res 
end


function _sdplr(
    pdata::ProblemData,
    algdata::AlgorithmData,
    config::Config,
)
    # TODO create all data structures

    # TODO double check scaling 

    # misc declarations
    recalcfreq = 5 
    recalc_cnt = 5 
    difficulty = 3 
    bestinfeas = 1.0e10

    # TODO setup printing
    if config.printlevel > 0
        printheading(1)
    end


    # set up algorithm parameters
    normb = norm(pdata.b, Inf)
    normC = C_normdatamat(pdata)
    best_dualbd = -1.0e20

    # initialize lbfgs datastructures
    lbfgshis = lbfgshistory(
        config.numlbfgsvecs,
        lbfgsvec[],
        0)

    for i = 1:config.numlbfgsvecs
        push!(lbfgshis.vecs, 
            lbfgsvec(zeros(size(algdata.R)), zeros(size(algdata.R)), 0.0, 0.0))
    end


    # TODO essential_calc
    ρ_c_tol = config.ρ_c / algdata.σ 
    val, ρ_c_val , ρ_f_val = 
        essential_calcs!(pdata, algdata, normC, normb)
    majiter = 0 
    iter = 0 # total number of iterations

    # save initial function value, notice that
    # here the constraints may not be satisfied,
    # which means the value may be smaller than the optimum
    origval = val 

    majoriter_end = false

    while majiter < config.maxmajiter 
        #avoid goto in C
        current_majoriter_end = false
        λ_update = 0
        while ((config.σ_strategy == 0 && λ_update < λ_updatect)
            ||(config.σ_strategy != 0 && difficulty != 1)) 

            # increase lambda counter, reset local iter counter and lastval
            λ_update += 1
            localiter = 0
            lastval = 1.0e10

            # check stopping criteria: rho_c_val = norm of gradient
            # once stationarity condition is satisfied, then break
            if ρ_c_val <= ρ_c_tol
                break
            end

            # in the local iteration, we keep optimizing
            # the subproblem using lbfgsb and return a solution
            # satisfying stationarity condition
            while (ρ_c_val > ρ_c_tol) 
                #increase both iter and localiter counters
                iter += 1
                localiter += 1
                # direction has been negated
                @show algdata.R
                @show algdata.G
                dir = dirlbfgs(algdata, lbfgshis, negate=true)

                @show sum(dir .* algdata.G)
                descent = sum(dir .* algdata.G)
                if isnan(descent) || descent >= 0 # not a descent direction
                    dir = -algdata.G # reverse back to gradient direction
                end

                @show dir
                lastval = val
                α, val = linesearch!(pdata, algdata, 
                                         dir, α_max=1.0, update=true) 

                algdata.R += α * dir
                @show algdata.R
                @show α
                if recalc_cnt == 0
                    val, ρ_c_val, ρ_f_val = 
                        essential_calcs!(pdata, algdata, normC, normb)
                    recalc_cnt = recalcfreq
                else
                    gradient!(pdata, algdata)
                    ρ_c_val = norm(algdata.G, 2) / (1.0 + normC)
                    ρ_f_val = norm(algdata.vio, 2) / (1.0 + normb)
                    recalc_cnt -= 1
                end
                @show ρ_c_val
                @show ρ_f_val
                @show algdata.vio

                if config.numlbfgsvecs > 0 
                    @show dir, α
                    lbfgs_postprocess!(algdata, lbfgshis, dir, α)
                end

                totaltime = time() - algdata.starttime

                if (totaltime >= config.timelimit 
                    || ρ_f_val <= config.ρ_f
                    ||  iter >= 10^7)
                    algdata.λ -= algdata.σ * algdata.vio
                    current_majoriter_end = true
                    break
                end

                bestinfeas = min(ρ_f_val, bestinfeas)
            end

            if current_majoriter_end
                majoriter_end = true
                break
            end

            # update Lagrange multipliers and recalculate essentials
            algdata.λ = -algdata.σ * algdata.vio
            val, ρ_c_val, ρ_f_val = 
                essential_calcs!(pdata, algdata, normC, normb)

            if config.σ_strategy == 1
                if localiter <= 10
                    difficulty = 1 # EASY
                elseif localiter > 10 && localiter <= 50 
                    difficulty = 2 # MEDIUM
                else
                    difficulty = 3 # HARD
                end
            end
            # TODO check dual bounds
        end # end one major iteration

        # cannot further improve infeasibility,
        # in other words to make the solution feasible, 
        # we get a ridiculously large value
        if val > 1.0e10 * abs(origval) 
            majoriter_end = true
            printf("Cannot reduce infeasibility any further.\n")
            break
        end

        if isnan(val)
            println("Error(sdplrlib): Got NaN.")
            return 0
        end

        if majoriter_end
            break
        end

        # TODO potential rank reduction 

        # update sigma
        while true
            algdata.σ *= config.σ_fac
            val, ρ_c_val, ρ_f_val = 
                essential_calcs!(pdata, algdata, normC, normb)
            ρ_c_tol = config.ρ_c / algdata.σ
            if ρ_c_tol < ρ_c_val
                break
            end
        end
        # refresh some parameters
        if config.σ_strategy == 1
            difficulty = 3
        end

        majiter += 1

        # clear bfgs vectors
        for i = 1:lbfgshis.m
            lbfgshis.vecs[i] = lbfgsvec(zeros(size(algdata.R)), zeros(size(algdata.R)), 0.0, 0.0)
        end
    end
end

using Test
using LinearAlgebra
using SparseArrays

A = [0 1;
     1 0]
n = size(A, 1)
d = sum(A, dims=2)[:, 1]
L = sparse(Diagonal(d) - A)
As = AbstractMatrix[]
bs = Float64[]
for i in eachindex(d)
    ei = zeros(n, 1)
    ei[i, 1] = 1
    push!(As, LowRankMatrix(Diagonal([1.0]), ei))
    push!(bs, 1.0)
end
r = 1
optval, optx = sdplr(-Float64.(L), As, bs, r)
optval == -4.0