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
    sparse_cons = SparseMatrixCSC{Tv, Ti}[]
    symlowrank_cons = SymLowRankMatrix{Tv}[]
    # treat diagonal matrices as sparse matrices
    sparse_As_global_inds = Ti[]
    symlowrank_As_global_inds = Ti[]
    
    # pre-allocate intermediate variables
    # for low-rank matrix evaluations
    BtVs = Matrix{Tv}[]
    BtUs = Matrix{Tv}[]
    Btvs = Vector{Tv}[]
    for (i, A) in enumerate(As)
        if isa(A, SparseMatrixCSC)
            push!(sparse_cons, A)
            push!(sparse_As_global_inds, i)
        elseif isa(A, Diagonal)
            push!(sparse_cons, sparse(A))
            push!(sparse_As_global_inds, i)
        elseif isa(A, SymLowRankMatrix)
            push!(symlowrank_cons, A)
            push!(symlowrank_As_global_inds, i)
            s = size(A.B, 2)
            # s and r are usually really small compared with n
            push!(BtVs, zeros(Tv, (s, r)))
            push!(BtUs, zeros(Tv, (s, r)))
            push!(Btvs, zeros(Tv, s))
        else
            @error "Currently only sparse/symmetric low-rank/diagonal constraints are supported."
        end
    end

    if isa(C, SparseMatrixCSC) 
        push!(sparse_cons, C)
        push!(sparse_As_global_inds, 0)
    elseif isa(C, Diagonal)
        push!(sparse_cons, sparse(C))
        push!(sparse_As_global_inds, 0)
    elseif isa(C, SymLowRankMatrix)
        push!(symlowrank_cons, C)
        push!(symlowrank_As_global_inds, 0)
        s = size(C.B, 2)
        # s and r are usually really small compared with n
        push!(BtVs, zeros(Tv, (s, r)))
        push!(BtUs, zeros(Tv, (s, r)))
        push!(Btvs, zeros(Tv, s))
    else
        @error "Currently only sparse/lowrank/diagonal objectives are supported."
    end

    triu_sum_A, agg_A_ptr, agg_A_nzind, agg_A_nzval_one, agg_A_nzval_two,
        sum_A, sum_A_to_triu_A_inds = preprocess_sparsecons(sparse_cons)
    
    n = size(C, 1)
    m = length(As)
    nnz_sum_A = length(sum_A.rowval)

    n = size(C, 1)
    # randomly initialize primal and dual variables
    R0 = 2 .* rand(n, r) .- 1
    λ0 = randn(m)

    # TODO: deal with indicies
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
                    length(symlowrank_cons),
                    symlowrank_cons, 
                    symlowrank_As_global_inds,
                    BtVs,
                    BtUs,
                    Btvs,
                    R0, 
                    zeros(Tv, size(R0)), 
                    λ0, 
                    zeros(Tv, m), 
                    zeros(Tv, m),
                    r,         #rank
                    10.0,      #sigma
                    zero(Tv),         #obj, will be initialized later
                    time(),           #starttime
                    zero(Tv),         #endtime 
                    zero(Tv),         #time spent on computing dual bound
                    zero(Tv),
                    Ti[],
                    Tv[],
                    Tv[],
                    zero(Tv),         #time spend on primal computation
                    zero(Tv),
                    )

    res = _sdplr(SDP, config)
    return res 
end


function _sdplr(
    SDP::SDPProblem{Ti, Tv, TC},
    config::BurerMonteiroConfig{Ti, Tv},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    n = size(SDP.R, 1)
    bestinfeas = 1.0e10
    SDP.starttime = time()
    lastprint = SDP.starttime # timestamp of last print
    R0 = deepcopy(SDP.R) 
    λ0 = deepcopy(SDP.λ)

    # TODO setup printing
    if config.printlevel > 0
        printheading(1)
    end


    # set up algorithm parameters
    normb = norm(SDP.b, 2)
    normC = norm(SDP.C, 2)
    best_dualbd = -1.0e20

    # initialize lbfgs datastructures
    lbfgshis = lbfgs_init(SDP.R, config.numlbfgsvecs)

    cur_gtol = 1.0 / SDP.σ     # stationarity tolerance
    cur_ptol = 1.0 / SDP.σ^0.1   # primal violation tolerance

    𝓛_val, stationarity_norm , primal_vio_norm = 
        essential_calcs!(SDP, normC, normb)
    iter = 0 # total number of iterations
    origval = 𝓛_val 

    dir = similar(SDP.R)
    majoriter = 0
    for _ = 1:config.maxmajoriter
        majoriter += 1
        localiter = 0
        while stationarity_norm > cur_gtol 
            # update iteration counters
            localiter += 1     
            iter += 1
            # find the lbfgs direction
            # the return direction has been negated
            lbfgs_dir!(dir, lbfgshis, SDP.G, negate=true)

            descent = dot(dir, SDP.G)
            if isnan(descent) || descent >= 0 # not a descent direction
                BLAS.scal!(-one(Tv), SDP.G)
                copyto!(dir, SDP.G) # reverse back to gradient direction
            end

            lastval = 𝓛_val # record last Lagrangian value
            # line search the best step size
            α ,𝓛_val = linesearch!(SDP, dir, α_max=1.0, update=true) 

            # update R and update gradient, stationarity, primal violence
            axpy!(α, dir, SDP.R)
            gradient!(SDP)
            stationarity_norm = norm(SDP.G, 2) / (1.0 + normC)
            primal_vio_norm = norm(SDP.primal_vio, 2) / (1.0 + normb)

            # if change of the Lagrangian value is small enough
            # then we terminate the current major iteration
            if (lastval - 𝓛_val) / max(1.0, abs(𝓛_val), lastval) < config.fprec * eps()
                break
            end
            # update lbfgs vectors
            if config.numlbfgsvecs > 0 
                lbfgs_update!(dir, lbfgshis, SDP.G, α)
            end

            current_time = time()
            if current_time - lastprint >= config.printfreq
                lastprint = current_time
                if config.printlevel > 0
                    printintermediate(majoriter, localiter, iter, 𝓛_val, 
                              SDP.obj, stationarity_norm, primal_vio_norm, best_dualbd)
                end
            end   

            if (current_time - SDP.starttime > config.maxtime
                || iter > config.maxiter)
                break
            end
        end


        printintermediate(majoriter, localiter, iter, 𝓛_val, 
                  SDP.obj, stationarity_norm, primal_vio_norm, best_dualbd)

        current_time = time()
        if current_time - SDP.starttime > config.maxtime
            @warn "Time limit exceeded. Stop optimizing."
            break
        end

        if iter > config.maxiter
            @warn "Iteration limit exceeded. Stop optimizing."
            break
        end


        if primal_vio_norm <= cur_ptol
            if primal_vio_norm <= config.ptol 
                eig_iter = Ti(ceil(2*max(iter, 1.0/config.objtol)^0.5*log(n))) 
                lanczos_dt, lanczos_eigval, GenericArpack_dt, GenericArpack_eigval, _, rel_duality_bound = surrogate_duality_gap(SDP, Tv(n), eig_iter;highprecision=true)  
                SDP.dual_lanczos_time += lanczos_dt
                SDP.dual_GenericArpack_time += GenericArpack_dt
                push!(SDP.checkdualbd_iters, iter)
                push!(SDP.lanczos_eigvals, lanczos_eigval)
                push!(SDP.GenericArpack_eigvals, GenericArpack_eigval)
                if rel_duality_bound <= config.objtol
                    @info "Duality gap and primal violence are small enough." primal_vio_norm rel_duality_bound stationarity_norm
                    break
                else
                    axpy!(-SDP.σ, SDP.primal_vio, SDP.λ)
                    cur_ptol = cur_ptol / SDP.σ^0.9
                    cur_gtol = cur_gtol / SDP.σ
                end
            else
                axpy!(-SDP.σ, SDP.primal_vio, SDP.λ)
                cur_ptol = cur_ptol / SDP.σ^0.9
                cur_gtol = cur_gtol / SDP.σ
            end
        else 
            SDP.σ *= config.σfac 
            cur_ptol = 1 / SDP.σ^0.1
            cur_gtol = 1 / SDP.σ 
        end

        cur_gtol = max(cur_gtol, config.gtol)
        cur_ptol = max(cur_ptol, config.ptol)

        𝓛_val, stationarity_norm, primal_vio_norm = 
            essential_calcs!(SDP, normC, normb)

        # clear lbfgs vectors for next major iteration
        for i = 1:lbfgshis.m
            lbfgshis.vecs[i] = LBFGSVector(zeros(size(SDP.R)), zeros(size(SDP.R)), zero(Tv), zero(Tv))
        end

        if majoriter == config.maxmajoriter
            @warn "Major iteration limit exceeded. Stop optimizing."
        end
    end
    
    𝓛_val, stationarity_norm, primal_vio_norm = essential_calcs!(SDP, normC, normb)
    println("Done")
    eig_iter = Ti(ceil(2*max(iter, 1.0/config.objtol)^0.5*log(n))) 
    lanczos_dt, lanczos_eigval, GenericArpack_dt, GenericArpack_eigval, duality_bound, rel_duality_bound = surrogate_duality_gap(SDP, Tv(n), eig_iter;highprecision=true)  
    SDP.dual_lanczos_time += lanczos_dt
    SDP.dual_GenericArpack_time += GenericArpack_dt
    push!(SDP.checkdualbd_iters, iter)
    push!(SDP.lanczos_eigvals, lanczos_eigval)
    push!(SDP.GenericArpack_eigvals, GenericArpack_eigval)

    SDP.endtime = time()
    totaltime = SDP.endtime - SDP.starttime
    SDP.primal_time = totaltime - SDP.dual_lanczos_time - SDP.dual_GenericArpack_time
    SDP.DIMACS_time = @elapsed begin
        DIMACS_errs = DIMACS_errors(SDP)
    end
    #@show normb, normC
    @show rel_duality_bound
    @show DIMACS_errs
    return Dict([
        "R" => SDP.R,
        "lambda" => SDP.λ,
        "R0" => R0,
        "lambda0" => λ0,
        "sigma" => SDP.σ,
        "stationarity" => stationarity_norm,
        "primal_vio" => primal_vio_norm,
        "obj" => SDP.obj,
        "duality_bound" => duality_bound,
        "rel_duality_bound" => rel_duality_bound,
        "totaltime" => totaltime,
        "dual_lanczos_time" => SDP.dual_lanczos_time,
        "dual_GenericArpack_time" => SDP.dual_GenericArpack_time,
        "checkdualbd_iters" => SDP.checkdualbd_iters,
        "lanczos_eigvals" => SDP.lanczos_eigvals,
        "GenericArpack_eigvals" => SDP.GenericArpack_eigvals,
        "primaltime" => SDP.primal_time,
        "iter" => iter,
        "majoriter" => majoriter,
        "DIMACS_errs" => DIMACS_errs,
        "gtol" => config.gtol,
        "ptol" => config.ptol,
        "dtol" => config.objtol,
    ])
end


