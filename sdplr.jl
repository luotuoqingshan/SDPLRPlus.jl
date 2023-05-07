using GenericArpack
include("structs.jl")
include("dataoper.jl")
include("lbfgs.jl")
include("myprint.jl")
include("linesearch.jl")
include("options.jl")

"""
Interface for SDPLR 

Problem formulation:
    min   Tr(C (YY^T))
    s.t.  Tr(As[i] (YY^T)) = bs[i]
            Y in R^{n x r}
"""
function sdplr(
    C::AbstractMatrix{Tv},
    As::Vector{Any},
    b::Vector{Tv},
    r::Ti;
    config::BurerMonteiroConfig{Ti, Tv}=BurerMonteiroConfig{Ti, Tv}(),
) where{Ti <: Integer, Tv <: AbstractFloat}
    m = length(As)
    #@assert (typeof(C) <: SparseMatrixCSC 
    #     || typeof(C) <: Diagonal
    #     || typeof(C) <: LowRankMatrix) "Wrong matrix type of cost matrix."
    #sparse_cons = SparseMatrixCSC{Tv, Ti}[]
    #dense_cons = Matrix{Tv}[]
    #diag_cons = Diagonal{Tv}[]
    #lowrank_cons = LowRankMatrix{Tv}[]
    #unitlowrank_cons = UnitLowRankMatrix{Tv}[]

    #sparse_bs = Tv[]
    #dense_bs = Tv[]
    #diag_bs = Tv[]
    #lowrank_bs = Tv[]
    #unitlowrank_bs = Tv[]
    #for i = 1:m
    #    if isa(As[i], SparseMatrixCSC)
    #        push!(sparse_cons, As[i])
    #        push!(sparse_bs, b[i])
    #    elseif isa(As[i], Diagonal) 
    #        push!(diag_cons, As[i])
    #        push!(diag_bs, b[i])
    #    elseif isa(As[i], Matrix) 
    #        push!(dense_cons, As[i])
    #        push!(dense_bs, b[i])
    #    elseif isa(As[i], LowRankMatrix) 
    #        push!(lowrank_cons, As[i])
    #        @show typeof(lowrank_bs)
    #        push!(lowrank_bs, b[i])
    #    elseif isa(As[i], UnitLowRankMatrix) 
    #        push!(unitlowrank_cons, As[i])
    #        push!(unitlowrank_bs, b[i])
    #    else
    #        error("Wrong matrix type of constraint matrix.")
    #    end
    #end
    #bs = [sparse_bs; dense_bs; diag_bs; lowrank_bs; unitlowrank_bs]
    #SDP = SDPProblem(m, sparse_cons, dense_cons, diag_cons, lowrank_cons,
    #                 unitlowrank_cons, C, bs)
    Constraints = Any[]
    for A in As
        if isa(A, LowRankMatrix)
            push!(Constraints, LowRankMatrix(A.D, A.B, r))
        elseif isa(A, UnitLowRankMatrix)
            push!(Constraints, LowRankMatrix(A.B, r))
        else
            push!(Constraints, A)
        end
    end
    SDP = SDPProblem(m, Constraints, C, b)
    n = size(C, 1)
    R_0 = 2 .* rand(n, r) .- 1
    BM = BurerMonteiro(
        R_0,              #R
        zeros(size(R_0)), #G, will be initialized later
        randn(m),         #λ
        zeros(m),         #vio(violation + obj), will be initialized later
        one(Tv) / n,          #σ
        zero(Tv),                #obj, will be initialized later
        time(),           #starttime
        zero(Tv),                #endtime 
        zero(Tv),                #time spent on computing dual bound
        zero(Tv),                #time spend on primal computation
    )
    res = _sdplr(BM, SDP, config)
    return res 
end


function _sdplr(
    BM::BurerMonteiro{Tv},
    SDP::SDPProblem{Ti, Tv, TC, TCons},
    config::BurerMonteiroConfig{Ti, Tv},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons}
    # misc declarations
    recalcfreq = 5 
    recalc_cnt = 5 
    difficulty = 3 
    bestinfeas = 1.0e10
    BM.starttime = time()
    lastprint = BM.starttime # timestamp of last print

    # TODO setup printing
    if config.printlevel > 0
        printheading(1)
    end


    # set up algorithm parameters
    normb = norm(SDP.b, Inf)
    normC = norm(SDP.C, Inf)
    best_dualbd = -1.0e20

    # initialize lbfgs datastructures
    lbfgshis = LBFGSHistory{Ti, Tv}(
        config.numlbfgsvecs,
        LBFGSVector{Tv}[],
        config.numlbfgsvecs::Ti)

    for i = 1:config.numlbfgsvecs
        push!(lbfgshis.vecs, 
            LBFGSVector(zeros(Tv, size(BM.R)),
                     zeros(Tv, size(BM.R)),
                     zero(Tv), zero(Tv)))
    end


    # TODO essential_calc
    tol_primal_vio = config.tol_primal_vio / BM.σ 
    𝓛_val, stationarity , primal_vio = 
        essential_calcs!(BM, SDP, normC, normb)
    majoriter = 0 
    iter = 0 # total number of iterations

    # save initial function value, notice that
    # here the constraints may not be satisfied,
    # which means the value may be smaller than the optimum
    origval = 𝓛_val 

    majoriter_end = false

    while majoriter < config.maxmajoriter 
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
            if stationarity <= config.tol_stationarity 
                break
            end

            # in the local iteration, we keep optimizing
            # the subproblem using lbfgsb and return a solution
            # satisfying stationarity condition
            while (stationarity > config.tol_stationarity) 
                #increase both iter and localiter counters
                iter += 1
                localiter += 1
                # direction has been negated
                dir = dirlbfgs(BM, lbfgshis, negate=true)

                descent = dot(dir, BM.G)
                if isnan(descent) || descent >= 0 # not a descent direction
                    dir .= -BM.G # reverse back to gradient direction
                end

                lastval = 𝓛_val
                α, 𝓛_val = linesearch!(BM, SDP, dir, α_max=1.0, update=true) 

                BM.R .+= α * dir
                if recalc_cnt == 0
                    𝓛_val, stationarity, primal_vio = 
                        essential_calcs!(BM, SDP, normC, normb)
                    recalc_cnt = recalcfreq
                else
                    gradient!(BM, SDP)
                    stationarity = norm(BM.G, 2) / (1.0 + normC)
                    primal_vio = norm(BM.primal_vio, 2) / (1.0 + normb)
                    recalc_cnt -= 1
                end

                if config.numlbfgsvecs > 0 
                    lbfgs_postprocess!(BM, lbfgshis, dir, α)
                end

                current_time = time() 
                if current_time - lastprint >= config.printfreq
                    lastprint = current_time
                    if config.printlevel > 0
                        printintermediate(majoriter, localiter, iter, 𝓛_val, 
                                  BM.obj, stationarity, primal_vio, best_dualbd)
                    end
                end   

                totaltime = time() - BM.starttime

                if (totaltime >= config.timelim 
                    || primal_vio <= config.tol_primal_vio
                    ||  iter >= 10^7)
                    @. BM.λ -= BM.σ * BM.primal_vio
                    current_majoriter_end = true
                    break
                end

                bestinfeas = min(primal_vio, bestinfeas)
            end

            if current_majoriter_end
                majoriter_end = true
                break
            end

            # update Lagrange multipliers and recalculate essentials
            @. BM.λ -= BM.σ * BM.primal_vio
            𝓛_val, stationarity, primal_vio = 
                essential_calcs!(BM, SDP, normC, normb)

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
        if 𝓛_val > 1.0e10 * abs(origval) 
            majoriter_end = true
            printf("Cannot reduce infeasibility any further.\n")
            break
        end

        if isnan(𝓛_val)
            println("Error(sdplrlib): Got NaN.")
            return 0
        end

        if majoriter_end
            break
        end

        # TODO potential rank reduction 

        # update sigma
        while true
            BM.σ *= config.σ_fac
            𝓛_val, stationarity, primal_vio = 
                essential_calcs!(BM, SDP, normC, normb)
            tol_stationarity = config.tol_stationarity / BM.σ
            if tol_stationarity < stationarity 
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
            lbfgshis.vecs[i] = LBFGSVector(zeros(size(BM.R)), zeros(size(BM.R)), 0.0, 0.0)
        end
    end
    𝓛_val, stationarity, primal_vio = essential_calcs!(BM, SDP, normC, normb)

    if config.checkdual
        BM.dual_time = @elapsed best_dualbd = dualbound(BM, SDP)
    end
    BM.endtime = time()
    totaltime = BM.endtime - BM.starttime
    BM.primal_time = totaltime - BM.dual_time
    return Dict([
        "R" => BM.R,
        "λ" => BM.λ,
        "stationarity" => stationarity,
        "primal_vio" => primal_vio,
        "obj" => BM.obj,
        "dualbd" => best_dualbd,
        "totattime" => totaltime,
        "dualtime" => BM.dual_time,
        "primaltime" => BM.primal_time,
    ])
end


function dualbound(
    BM::BurerMonteiro,
    SDP::SDPProblem, 
)
    n = size(BM.R, 1)
    op = ArpackSimpleFunctionOp(
        (y, x) -> begin
                mul!(y, SDP.C, x)
                for i = 1:SDP.m
                     y .-= BM.λ[i] * (SDP.constraints[i] * x) 
                end
                return y
        end, n)
    eigenvals, eigenvecs = symeigs(op, 1; which=:SA, ncv=min(100, n), maxiter=1000000)
    dualbound = real.(eigenvals[1]) 
    return dualbound 
end


