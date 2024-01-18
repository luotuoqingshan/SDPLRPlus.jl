"""
Interface for SDPLR 

Problem formulation:
    min   Tr(C (YY^T))
    s.t.  Tr(As[i] (YY^T)) = bsi]
            Y in R^{n x r}
"""
function sdplr(
    C::AbstractMatrix{Tv},
    As::Vector{Any},
    b::Vector{Tv},
    r::Ti;
    config::BurerMonteiroConfig{Ti, Tv}=BurerMonteiroConfig{Ti, Tv}(),
) where{Ti <: Integer, Tv <: AbstractFloat}
    sparse_cons = SparseMatrixCSC{Tv, Ti}[]
    # treat diagonal matrices as sparse matrices
    sparse_As_global_inds = Ti[]
    for (i, A) in enumerate(As)
        if isa(A, SparseMatrixCSC)
            push!(sparse_cons, A)
            push!(sparse_As_global_inds, i)
        elseif isa(A, Diagonal)
            push!(sparse_cons, sparse(A))
            push!(sparse_As_global_inds, i)
        end
    end

    if isa(C, SparseMatrixCSC) 
        push!(sparse_cons, C)
        push!(sparse_As_global_inds, 0)
    elseif isa(C, Diagonal)
        push!(sparse_cons, sparse(C))
        push!(sparse_As_global_inds, 0)
    end

    triu_sum_A, agg_A_ptr, agg_A_nzind, agg_A_nzval_one, agg_A_nzval_two,
        sum_A, sum_A_to_triu_A_inds = preprocess_sparsecons(sparse_cons)
    
    n = size(C, 1)
    m = length(As)
    nnz_sum_A = length(sum_A.rowval)

    n = size(C, 1)
    # randomly initialize primal and dual variables
    R₀ = 2 .* rand(n, r) .- 1
    λ₀ = randn(m)

    SDP = SDPProblem(n, m, # size of X, number of constraints
                     C, # objective matrix
                     b, # right-hand-side vector of constraints 
                     triu_sum_A.colptr, 
                     triu_sum_A.rowval, # (colptr, rowval) for aggregated triu(A)
                     length(sparse_cons), 
                     agg_A_ptr, 
                     agg_A_nzind,
                     agg_A_nzval_one, 
                     agg_A_nzval_two, 
                     sparse_As_global_inds,
                     zeros(Tv, nnz_sum_A), 
                     zeros(Tv, nnz_sum_A),
                     sum_A,
                     sum_A_to_triu_A_inds, 
                     zeros(Tv, nnz_sum_A), 
                     zeros(Tv, m), zeros(Tv, m),
                     R₀, 
                     zeros(Tv, size(R₀)), 
                     λ₀, 
                     zeros(Tv, m), 
                     zeros(Tv, m),
                    BurerMonterioMutableScalars(
                        r, 
                        one(Tv) / n,      #σ
                        zero(Tv),         #obj, will be initialized later
                        time(),           #starttime
                        zero(Tv),         #endtime 
                        zero(Tv),         #time spent on computing dual bound
                        zero(Tv),         #time spend on primal computation
                    ))

    res = _sdplr(SDP, config)
    return res 
end


function _sdplr(
    SDP::SDPProblem{Ti, Tv, TC},
    config::BurerMonteiroConfig{Ti, Tv},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    # misc declarations
    recalcfreq = 5 
    recalc_cnt = 10^7 
    difficulty = 3 
    bestinfeas = 1.0e10
    SDP.scalars.starttime = time()
    lastprint = SDP.scalars.starttime # timestamp of last print
    R₀ = deepcopy(SDP.R) 
    λ₀ = deepcopy(SDP.λ)

    #@show size(R₀)
    #println("haha")
    #include("readdata.jl")
    #write_initial_solution(R₀, λ₀, "G1.solin")

    # TODO setup printing
    if config.printlevel > 0
        printheading(1)
    end


    # set up algorithm parameters
    normb = norm(SDP.b, Inf)
    #normC = norm(SDP.C, Inf)
    normC = maximum(abs.(SDP.C))
    best_dualbd = -1.0e20

    # initialize lbfgs datastructures
    lbfgshis = LBFGSHistory{Ti, Tv}(
        config.numlbfgsvecs,
        LBFGSVector{Tv}[],
        Ref(config.numlbfgsvecs))

    for i = 1:config.numlbfgsvecs
        push!(lbfgshis.vecs, 
            LBFGSVector(zeros(Tv, size(SDP.R)),
                        zeros(Tv, size(SDP.R)),
                        Ref(zero(Tv)), 
                        Ref(zero(Tv)),
                        ))
    end


    tol_stationarity = config.tol_stationarity / SDP.scalars.σ 

    𝓛_val, stationarity , primal_vio = 
        essential_calcs!(SDP, normC, normb)
    majoriter = 0 
    iter = 0 # total number of iterations


    # save initial function value, notice that
    # here the constraints may not be satisfied,
    # which means the value may be smaller than the optimum
    origval = 𝓛_val 

    majoriter_end = false
    dir = similar(SDP.R)

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
            if stationarity <= tol_stationarity 
                break
            end

            # in the local iteration, we keep optimizing
            # the subproblem using lbfgsb and return a solution
            # satisfying stationarity condition
            while (stationarity > tol_stationarity) 
                #@show tol_stationarity
                #increase both iter and localiter counters
                iter += 1
                localiter += 1
                # direction has been negated
                dirlbfgs_dt = @elapsed begin
                    dirlbfgs!(dir, SDP, lbfgshis, negate=true)
                end
                #@show dirlbfgs_dt
                #@show norm(dir)

                descent = dot(dir, SDP.G)
                if isnan(descent) || descent >= 0 # not a descent direction
                    LinearAlgebra.BLAS.scal!(-one(Tv), SDP.G)
                    #lmul!(-one(Tv), SDP.G)
                    copyto!(dir, SDP.G) # reverse back to gradient direction
                end

                lastval = 𝓛_val
                linesearch_dt = @elapsed begin
                    α ,𝓛_val = linesearch!(SDP, dir, α_max=1.0, update=true) 
                end
                #@printf("Iter %d\n", iter)
                #@show linesearch_dt
                #@printf("iter %d, 𝓛_val %.10lf α %.10lf\n", iter, 𝓛_val, α) 
                #@show iter, 𝓛_val

                LinearAlgebra.axpy!(α, dir, SDP.R)
                if recalc_cnt == 0
                    𝓛_val, stationarity, primal_vio = 
                        essential_calcs!(SDP, normC, normb)
                    recalc_cnt = recalcfreq
                    #@show 𝓛_val, stationarity, primal_vio
                else
                    gradient!(SDP)
                    stationarity = norm(SDP.G, 2) / (1.0 + normC)
                    primal_vio = norm(SDP.primal_vio, 2) / (1.0 + normb)
                    recalc_cnt -= 1
                end

                #@show stationarity, primal_vio

                lbfgs_postprecess_dt = @elapsed begin
                    if config.numlbfgsvecs > 0 
                        lbfgs_postprocess!(SDP, lbfgshis, dir, α)
                    end
                end
                #@show lbfgs_postprecess_dt

                current_time = time() 
                if current_time - lastprint >= config.printfreq
                    lastprint = current_time
                    if config.printlevel > 0
                        printintermediate(majoriter, localiter, iter, 𝓛_val, 
                                  SDP.scalars.obj, stationarity, primal_vio, best_dualbd)
                    end
                end   

                totaltime = time() - SDP.scalars.starttime

                if (totaltime >= config.timelim 
                    || primal_vio <= config.tol_primal_vio
                    ||  iter >= 10^7)
                    LinearAlgebra.axpy!(-SDP.scalars.σ, SDP.primal_vio, SDP.λ)
                    current_majoriter_end = true
                    break
                end
                bestinfeas = min(primal_vio, bestinfeas)
            end # end of local iteration

            if current_majoriter_end
                printintermediate(majoriter, localiter, iter, 𝓛_val, 
                          SDP.scalars.obj, stationarity, primal_vio, best_dualbd)
                majoriter_end = true
                break
            end

            # update Lagrange multipliers and recalculate essentials
            LinearAlgebra.axpy!(-SDP.scalars.σ, SDP.primal_vio, SDP.λ)
            𝓛_val, stationarity, primal_vio = 
                essential_calcs!(SDP, normC, normb)

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
        if config.checkdual
            SDP.scalars.dual_time += @elapsed begin            
                
            end
        end

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
            SDP.scalars.σ *= config.σ_fac
            𝓛_val, stationarity, primal_vio = 
                essential_calcs!(SDP, normC, normb)
            tol_stationarity = config.tol_stationarity / SDP.scalars.σ
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
            lbfgshis.vecs[i] = LBFGSVector(zeros(size(SDP.R)), zeros(size(SDP.R)), Ref(0.0), Ref(0.0))
        end
    end
    𝓛_val, stationarity, primal_vio = essential_calcs!(SDP, normC, normb)
    println("Done")
    if config.checkdual
        SDP.scalars.dual_time = @elapsed best_dualbd = dualbound(SDP)
    end
    SDP.scalars.endtime = time()
    totaltime = SDP.scalars.endtime - SDP.scalars.starttime
    SDP.scalars.primal_time = totaltime - SDP.scalars.dual_time
    #@show normb, normC
    return Dict([
        "R" => SDP.R,
        "λ" => SDP.λ,
        "R₀" => R₀,
        "λ₀" => λ₀,
        "σ" => SDP.scalars.σ,
        "stationarity" => stationarity,
        "primal_vio" => primal_vio,
        "obj" => SDP.scalars.obj,
        "dualbd" => best_dualbd,
        "totattime" => totaltime,
        "dualtime" => SDP.scalars.dual_time,
        "primaltime" => SDP.scalars.primal_time,
        "iter" => iter,
        "majoriter" => majoriter,
    ])
end



function surrogate_duality_gap(
    SDP::SDPProblem, 
)
    essential_calcs!()
    g_t = SDP.scalars.obj  
    duality_gap = (g_t - dot(SDP.λ, SDP.primal_vio) - 
                   SDP.scalars.σ/2*dot(SDP.primal_vio, SDP.primal_vio))
    n = size(SDP.R, 1)
    op = ArpackSimpleFunctionOp(
        (y, x) -> begin
                mul!(y, SDP.C, x)
                for i = 1:SDP.m
                     y .-= SDP.λ[i] * (SDP.constraints[i] * x) 
                end
                return y
        end, n)
    eigenvals, eigenvecs = symeigs(op, 1; which=:SA, ncv=min(100, n), maxiter=1000000)
    dualbound = real.(eigenvals[1]) 
    return dualbound 
end

