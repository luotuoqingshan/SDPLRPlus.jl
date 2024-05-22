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
    kwargs...
) where{Ti <: Integer, Tv}
    for (key, value) in kwargs
        if hasfield(BurerMonteiroConfig, Symbol(key))
            setfield!(config, Symbol(key), value)
        else
            @warn "Ignoring unrecognized keyword argument $key"
        end
    end 

    if config.printlevel > 0
        printheading(1)
    end

    preprocess_dt = @elapsed begin
        m = length(As)
        (sparse_cons = 
            Union{SparseMatrixCSC{Tv, Ti}, SparseMatrixCOO{Tv, Ti}}[])
        symlowrank_cons = SymLowRankMatrix{Tv}[]
        # treat diagonal matrices as sparse matrices
        sparse_As_global_inds = Ti[]
        symlowrank_As_global_inds = Ti[]
    
        for (i, A) in enumerate(As)
            if isa(A, Union{SparseMatrixCSC, SparseMatrixCOO})
                push!(sparse_cons, A)
                push!(sparse_As_global_inds, i)
            elseif isa(A, Diagonal)
                push!(sparse_cons, sparse(A))
                push!(sparse_As_global_inds, i)
            elseif isa(A, SymLowRankMatrix)
                push!(symlowrank_cons, A)
                push!(symlowrank_As_global_inds, i)
            else
                @error "Currently only sparse\
                /symmetric low-rank\
                /diagonal constraints are supported."
            end
        end

        if isa(C, Union{SparseMatrixCSC, SparseMatrixCOO}) 
            push!(sparse_cons, C)
            push!(sparse_As_global_inds, m+1)
        elseif isa(C, Diagonal)
            push!(sparse_cons, sparse(C))
            push!(sparse_As_global_inds, m+1)
        elseif isa(C, SymLowRankMatrix)
            push!(symlowrank_cons, C)
            push!(symlowrank_As_global_inds, m+1)
        else
            @error "Currently only sparse\
            /lowrank/diagonal objectives are supported."
        end
        @info "Finish classifying constraints."
        res = @timed begin
            triu_agg_sparse_A, triu_agg_sparse_A_matptr, 
            triu_agg_sparse_A_nzind, triu_agg_sparse_A_nzval_one, 
            triu_agg_sparse_A_nzval_two, agg_sparse_A, 
            agg_sparse_A_mappedto_triu = preprocess_sparsecons(sparse_cons)
        end
        @debug "$(res.bytes)B allocated during preprocessing constraints." 
    
        n = size(C, 1)
        nnz_triu_agg_sparse_A = length(triu_agg_sparse_A.rowval)

        # randomly initialize primal and dual variables
        Rt0 = 2 .* rand(r, n) .- 1
        λ0 = randn(m)

        data = SDPData(n, m, C, As, b)
        var = SolverVars(
            Rt0,
            zeros(Tv, size(Rt0)),
            λ0,
            Ref(r),
            Ref(2.0),
            Ref(zero(Tv)),
        )
        aux = SolverAuxiliary(
            length(sparse_cons),
            triu_agg_sparse_A_matptr,
            triu_agg_sparse_A_nzind,
            triu_agg_sparse_A_nzval_one,
            triu_agg_sparse_A_nzval_two,
            agg_sparse_A_mappedto_triu,
            sparse_As_global_inds,

            triu_agg_sparse_A,
            agg_sparse_A,
            zeros(Tv, nnz_triu_agg_sparse_A), # UVt
            zeros(Tv, m+1), zeros(Tv, m+1), # A_RD, A_DD

            length(symlowrank_cons), #n_symlowrank_matrices
            symlowrank_cons, 
            symlowrank_As_global_inds,

            zeros(Tv, m+1), # y, auxiliary variable for 𝒜t 
            zeros(Tv, m+1), # primal_vio
        )
        stats = SolverStats(
            Ref(zero(Tv)), # starttime
            Ref(zero(Tv)), # endtime
            Ref(zero(Tv)), # time spent on lanczos with random start
            Ref(zero(Tv)), # time spent on GenericArpack
            Ti[],
            Tv[],
            Tv[],
            Ref(zero(Tv)), # primal time
            Ref(zero(Tv)), # DIMACS time
        )
    end
    @debug "preprocess dt" preprocess_dt
    ans = _sdplr(data, var, aux, stats, config)
    ans["preprocess_time"] = preprocess_dt
    ans["totaltime"] += preprocess_dt
    return ans 
end


function _sdplr(
    data::SDPData{Ti, Tv, TC},
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
    stats::SolverStats{Ti, Tv},
    config::BurerMonteiroConfig{Ti, Tv},
) where{Ti <: Integer, Tv, TC <: AbstractMatrix{Tv}}
    n = data.n # size of decision variables 
    m = data.m # number of constraints
    stats.starttime[] = time()
    lastprint = stats.starttime[] # timestamp of last print
    Rt0 = deepcopy(var.Rt) 
    λ0 = deepcopy(var.λ)

    # set up algorithm parameters
    normb = norm(data.b, 2)
    normC = norm(data.C, 2)
    best_dualbd = -1.0e20

    # initialize lbfgs datastructures
    lbfgshis = lbfgs_init(var.Rt, config.numlbfgsvecs)

    cur_gtol = 1.0 / var.σ[]     # stationarity tolerance
    cur_ptol = 1.0 / var.σ[]^0.1   # primal violation tolerance

    𝓛_val, grad_norm, primal_vio_norm = fg!(data, var, aux, normC, normb)
    iter = 0 # total number of iterations

    dirt = similar(var.Rt)
    majoriter = 0


    rankupd_tol_cnt = config.rankupd_tol
    min_rel_duality_gap = 1e20
    for _ = 1:config.maxmajoriter
        majoriter += 1
        localiter = 0

        #
        #
        #
        while grad_norm > cur_gtol 
            # update iteration counters
            localiter += 1     
            iter += 1
            # find the lbfgs direction
            # the return direction has been negated
            lbfgs_dir_dt = @elapsed begin
                lbfgs_dir!(dirt, lbfgshis, var.Gt, negate=true)
            end
            @debug "lbfgs dir dt" lbfgs_dir_dt

            descent = dot(dirt, var.Gt)
            if isnan(descent) || descent >= 0 # not a descent direction
                BLAS.scal!(-one(Tv), var.Gt)
                copyto!(dirt, var.Gt) # reverse back to gradient direction
            end

            lastval = 𝓛_val # record last Lagrangian value
            # line search the best step size
            linesearch_dt = @elapsed begin
                α ,𝓛_val = linesearch!(var, aux, dirt, α_max=1.0) 
            end
            @debug "line search time" linesearch_dt

            # update R and update gradient, stationarity, primal violence
            axpy!(α, dirt, var.Rt)
            g_dt = @elapsed begin
                g!(var, aux)
            end
            @debug "g time" g_dt
            grad_norm = norm(var.Gt, 2) / (1.0 + normC)
            v = @view aux.primal_vio[1:m]
            primal_vio_norm = norm(v, 2) / (1.0 + normb)

            # if change of the Lagrangian value is small enough
            # then we terminate the current major iteration
            rel_delta = (lastval - 𝓛_val) / max(1.0, abs(𝓛_val), abs(lastval))
            if rel_delta < config.fprec * eps()
                break
            end
            # update lbfgs vectors
            if config.numlbfgsvecs > 0 
                lbfgs_update!(dirt, lbfgshis, var.Gt, α)
            end

            current_time = time()
            if current_time - lastprint >= config.printfreq
                lastprint = current_time
                if config.printlevel > 0
                    printintermediate(config.dataset, majoriter, 
                              localiter, iter, 𝓛_val, var.obj[], grad_norm,
                              primal_vio_norm, best_dualbd)
                end
            end   

            if (current_time - stats.starttime[] > config.maxtime
                || iter > config.maxiter)
                break
            end
        end


        current_time = time()
        printintermediate(config.dataset, majoriter, localiter, iter, 𝓛_val, 
                  var.obj[], grad_norm, primal_vio_norm, best_dualbd)
        lastprint = current_time

        if current_time - stats.starttime[] > config.maxtime
            @warn "Time limit exceeded. Stop optimizing."
            break
        end

        if iter > config.maxiter
            @warn "Iteration limit exceeded. Stop optimizing."
            break
        end

        rank_double = false

        if primal_vio_norm <= cur_ptol
            if primal_vio_norm <= config.ptol 
                @info "primal vio is small enough, checking duality bound."
                eig_iter = Ti(2*ceil(max(iter, 100)^0.5*log(n))) 
                lanczos_dt, lanczos_eigval, GenericArpack_dt, 
                GenericArpack_eigval, _, rel_duality_bound = 
                    surrogate_duality_gap(data, var, aux, 
                    config.prior_trace_bound, eig_iter;highprecision=false)  
                stats.dual_lanczos_time[] += lanczos_dt
                stats.dual_GenericArpack_time[] += GenericArpack_dt
                push!(stats.checkdualbd_iters, iter)
                push!(stats.lanczos_eigvals, lanczos_eigval)
                push!(stats.GenericArpack_eigvals, GenericArpack_eigval)
                @info "rel_duality_bound" rel_duality_bound
                if rel_duality_bound <= config.objtol
                    @info "Duality gap and primal violence are small enough." 
                    @info  primal_vio_norm rel_duality_bound grad_norm
                    break
                else
                    if min_rel_duality_gap - rel_duality_bound < config.objtol
                        rankupd_tol_cnt -= 1
                    else
                        rankupd_tol_cnt = config.rankupd_tol
                    end
                    (min_rel_duality_gap = 
                        min(min_rel_duality_gap, rel_duality_bound))
                    if rankupd_tol_cnt == 0
                        rank_double = true
                    end
                    #last_rel_duality_bound = rel_duality_bound
                    v = @view aux.primal_vio[1:m]
                    axpy!(-var.σ[], v, var.λ)
                    cur_ptol = cur_ptol / var.σ[]^0.9
                    cur_gtol = cur_gtol / var.σ[]
                end
            else
                v = @view aux.primal_vio[1:m]
                axpy!(-var.σ[], v, var.λ)
                cur_ptol = cur_ptol / var.σ[]^0.9
                cur_gtol = cur_gtol / var.σ[]
            end
        else 
            var.σ[] *= config.σfac 
            cur_ptol = 1 / var.σ[]^0.1
            cur_gtol = 1 / var.σ[] 
        end

        #cur_ptol = max(cur_ptol, config.ptol)
        #@info "cur_ptol, cur_gtol:" cur_ptol, cur_gtol

        # when objective gap doesn't improve, we double the rank
        if rank_double 
            var = rank_update!(var)
            cur_ptol = 1 / var.σ[]^0.1
            cur_gtol = 1 / var.σ[]
            lbfgshis = lbfgs_init(var.Rt, config.numlbfgsvecs)
            dirt = similar(var.Rt)
            min_rel_duality_gap = 1e20
            rankupd_tol_cnt = config.rankupd_tol
            @info "rank doubled, newrank is $(size(var.Rt, 1))."
        else
            lbfgs_clear!(lbfgshis)
        end

        𝓛_val, grad_norm, primal_vio_norm = fg!(data, var, aux, normC, normb)

        if majoriter == config.maxmajoriter
            @warn "Major iteration limit exceeded. Stop optimizing."
        end
    end
    
    𝓛_val, grad_norm, primal_vio_norm = fg!(data, var, aux, normC, normb)
    println("Done")
    eig_iter = Ti(ceil(2*max(iter, 100)^0.5*log(n))) 

    lanczos_dt, lanczos_eigval, GenericArpack_dt, 
        GenericArpack_eigval, duality_bound, 
        rel_duality_bound = surrogate_duality_gap(
            data, var, aux, config.prior_trace_bound, eig_iter;
            highprecision=false)

    stats.dual_lanczos_time[] += lanczos_dt
    stats.dual_GenericArpack_time[] += GenericArpack_dt
    push!(stats.checkdualbd_iters, iter)
    push!(stats.lanczos_eigvals, lanczos_eigval)
    push!(stats.GenericArpack_eigvals, GenericArpack_eigval)

    stats.endtime[] = time()
    totaltime = stats.endtime[] - stats.starttime[]
    (stats.primal_time[] = 
        totaltime - stats.dual_lanczos_time[] - stats.dual_GenericArpack_time[])
    stats.DIMACS_time[] = @elapsed begin
        if config.eval_DIMACS_errs
            DIMACS_errs = DIMACS_errors(data, var, aux)
        else
            DIMACS_errs = zeros(6)
        end
    end
    #@show normb, normC
    @show rel_duality_bound
    @show DIMACS_errs
    return Dict([
        "Rt" => var.Rt,
        "lambda" => var.λ,
        "Rt0" => Rt0,
        "lambda0" => λ0,
        "sigma" => var.σ[],
        "grad_norm" => grad_norm,
        "primal_vio" => primal_vio_norm,
        "obj" => var.obj[],
        "duality_bound" => duality_bound,
        "rel_duality_bound" => rel_duality_bound,
        "totaltime" => totaltime,
        "dual_lanczos_time" => stats.dual_lanczos_time[],
        "dual_GenericArpack_time" => stats.dual_GenericArpack_time[],
        "checkdualbd_iters" => stats.checkdualbd_iters,
        "lanczos_eigvals" => stats.lanczos_eigvals,
        "GenericArpack_eigvals" => stats.GenericArpack_eigvals,
        "primaltime" => stats.primal_time[],
        "iter" => iter,
        "majoriter" => majoriter,
        "DIMACS_errs" => DIMACS_errs,
        "ptol" => config.ptol,
        "objtol" => config.objtol,
        "fprec" => config.fprec,
        "rankupd_tol" => config.rankupd_tol,
        "r" => size(var.Rt, 1),
    ])
end
