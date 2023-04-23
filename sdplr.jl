using GenericArpack
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
    timelim = 3600.0, # seconds
    printlevel = 1,
    printfreq = 60.0, # seconds
    numblbfgsvec = 3, 
    σ_strategy = 1,
    λ_updatect = 1,
    majoriterlim = 10^5,
    iterlim = 10^7,
    checkdual = true,
)
    m = length(As)
    @assert (typeof(C) <: SparseMatrixCSC 
         || typeof(C) <: Diagonal
         || typeof(C) <: LowRankMatrix) "Wrong matrix type of cost matrix."
    A_sp = SparseMatrixCSC[]
    A_diag = Diagonal[]
    A_lr = LowRankMatrix[]
    A_dense = Matrix[]
    for i = 1:m
        if typeof(As[i]) <: SparseMatrixCSC
            push!(A_sp, As[i])
        elseif typeof(As[i]) <: Diagonal
            push!(A_d, As[i])
        elseif typeof(As[i]) <: LowRankMatrix
            push!(A_lr, As[i])
        elseif typeof(As[i]) <: Matrix
            push!(A_dense, As[i])
        else
            error("Wrong matrix type of constraint matrix.")
        end
    end
    pdata = ProblemData(m, length(A_sp), length(A_diag), length(A_lr),
                        length(A_dense), A_sp, A_diag, A_lr, A_dense, C, bs)
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
        0,                #endtime 
        0,                #time spent on computing dual bound
        0,                #time spend on primal computation
    )
    config = Config(ρ_f, ρ_c, σ_fac, timelim, printlevel, printfreq, 
                    numblbfgsvec, σ_strategy, λ_updatect, majoriterlim,
                    iterlim, checkdual)
    res = _sdplr(pdata, algdata, config)
    return res 
end


function _sdplr(
    pdata::ProblemData,
    algdata::AlgorithmData,
    config::Config,
)
    # misc declarations
    recalcfreq = 5 
    recalc_cnt = 5 
    difficulty = 3 
    bestinfeas = 1.0e10
    algdata.starttime = time()
    lastprint = algdata.starttime # timestamp of last print

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
    majoriter = 0 
    iter = 0 # total number of iterations

    # save initial function value, notice that
    # here the constraints may not be satisfied,
    # which means the value may be smaller than the optimum
    origval = val 

    majoriter_end = false

    while majoriter < config.majoriterlim 
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
                dir = dirlbfgs(algdata, lbfgshis, negate=true)

                descent = sum(dir .* algdata.G)
                if isnan(descent) || descent >= 0 # not a descent direction
                    dir = -algdata.G # reverse back to gradient direction
                end

                lastval = val
                α, val = linesearch!(pdata, algdata, 
                                         dir, α_max=1.0, update=true) 

                algdata.R += α * dir
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

                if config.numlbfgsvecs > 0 
                    lbfgs_postprocess!(algdata, lbfgshis, dir, α)
                end

                current_time = time() 
                if current_time - lastprint >= config.printfreq
                    lastprint = current_time
                    if config.printlevel > 0
                        printintermediate(majoriter, localiter, iter, val, 
                                  algdata.obj, ρ_c_val, ρ_f_val, best_dualbd)
                    end
                end   

                totaltime = time() - algdata.starttime

                if (totaltime >= config.timelim 
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
        end # end one major iteration

        # TODO check dual bounds

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
        λ_update = 0
        if config.σ_strategy == 1
            difficulty = 3
        end

        majoriter += 1

        # clear bfgs vectors
        for i = 1:lbfgshis.m
            lbfgshis.vecs[i] = lbfgsvec(zeros(size(algdata.R)), zeros(size(algdata.R)), 0.0, 0.0)
        end
    end
    val, ρ_c_val, ρ_f_val = essential_calcs!(pdata, algdata, normC, normb)

    if config.checkdual
        algdata.dualcalctime = @elapsed best_dualbd = dualbound(pdata, algdata)
    end
    algdata.endtime = time()
    totaltime = algdata.endtime - algdata.starttime
    algdata.primaltime = totaltime - algdata.dualcalctime
    return Dict([
        "R" => algdata.R,
        "λ" => algdata.λ,
        "ρ_c_val" => ρ_c_val,
        "ρ_f_val" => ρ_f_val,
        "obj" => algdata.obj,
        "dualbd" => best_dualbd,
        "totattime" => totaltime,
        "dualtime" => algdata.dualcalctime,
        "primaltime" => algdata.primaltime,
    ])
end


function dualbound(
    pdata::ProblemData, 
    algdata::AlgorithmData
)
    n = size(algdata.R, 1)
    op = ArpackSimpleFunctionOp(
        (y, x) -> begin
                mul!(y, pdata.C, x)
                for i = 1:pdata.m
                    if i <= pdata.m_sp
                        y .-= algdata.λ[i] * (pdata.A_sp[i] * x)
                    elseif i <= pdata.m_sp + pdata.m_diag
                        j = i - pdata.m_sp
                        y .-= algdata.λ[i] * (pdata.A_diag[j] * x)
                    elseif i <= pdata.m_sp + pdata.m_diag + pdata.m_lr
                        j = i - pdata.m_sp - pdata.m_diag
                        # BDBᵀ x
                        y .-= algdata.λ[i] * pdata.A_lr[i].B * 
                            (pdata.A_lr[j].D * (pdata.A_lr[j].B' * x))
                    else 
                        j = i - pdata.m_sp - pdata.m_diag - pdata.m_lr
                        y .-= algdata.λ[i] * (pdata.A_dense[j] * x)
                    end
                end
                return y
        end, n)
    eigenvals, eigenvecs = symeigs(op, 1; which=:SA, ncv=min(100, n), maxiter=1000000)
    dualbound = real.(eigenvals[1]) 
    return dualbound 
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
res = sdplr(-Float64.(L), As, bs, r)
@show res