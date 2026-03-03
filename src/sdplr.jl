@doc raw"""
    sdplr(C, As, b, r)
    sdplr(C, As, b, r; kwargs...)

These functions tackle the following semidefinite program
```math
\begin{aligned}
    \text{minimize}_{X \succeq 0} \quad    &\langle C , X \rangle \\
    \text{subject to}\quad  &\langle A_i, X \rangle = b_i, \quad \forall i \in [m_{\text{eq}}]\\
                            &\langle A_i, X \rangle \leq b_i, \quad \forall i \in [m_{\text{ineq}}]\\
                &   X \in \mathbb{R}^{n \times n}
\end{aligned}
```
by factorizing the solution matrix ``X`` as ``YY^T`` and solve the following
nonlinear program instead.
```math
\begin{aligned}
    \text{minimize}\quad    &\langle C , YY^T \rangle \\
    \text{subject to}\quad  &\langle A_i, YY^T \rangle = b_i, \quad \forall i \in [m_{\text{eq}}]\\
                            &\langle A_i, YY^T \rangle \leq b_i, \quad \forall i \in [m_{\text{ineq}}]\\
                &   Y \in \mathbb{R}^{n \times r}
\end{aligned}
```
Inequality constraints are handled via Armijo line search instead of the exact
quartic line search used for equality-only problems.

Arguments
---------
- `As` is a vector of ``m`` constraint matrices ``A_i`` of size ``n \times n``. There are four types of constraint matrices supported:
    * `SparseMatrixCSC` for sparse constraints with nnz ``\Theta (n)``.
    * `SparseMatrixCOO` for super sparse constraints with nnz ``o(n)``.
    * `SymLowRankMatrix` for low-rank constraints with form ``BDB^T``.
    * `Diagonal` for diagonal constraints. Consider using `SparseMatrixCOO` instead if the diagonal matrix is super sparse.
- `C` is the cost matrix ``C`` of size ``n \times n``. Currently we support four types mentioned above.
- `b` is a vector of m right-hand side values ``b_i``.
- `r` is the initial rank of the solution matrix ``Y``.
- `constraint_types`: Optional `AbstractVector{Bool}` of length ``m`` indicating
    whether each constraint is an inequality (``\leq``). `true` means ``\leq``,
    `false` means ``=``. If not provided, all constraints are treated as equalities.

Optional arguments
------------------
- `ptol`: Primal infeasibility tolerance. Interpretation depends on `ptol_mode`.
    The default value is ``10^{-2}``.
- `ptol_mode`: Controls how primal infeasibility is measured.
    * `:relative` (default): ``\|\mathcal{A}(YY^T) - b\|_2 / \|b\|_2``.
    * `:absolute`: ``\|\mathcal{A}(YY^T) - b\|_2``.
- `objtol`: Duality gap tolerance. Interpretation depends on `objtol_mode`.
    The default value is ``10^{-2}``. Set to `Inf` to skip the duality gap check.
- `objtol_mode`: Controls how the duality gap is measured.
    * `:relative` (default): ``(\langle C, YY^T\rangle - d^*) / \min(|\langle C, YY^T\rangle|, |d^*|)``,
      where ``d^*`` is the best dual bound found.
    * `:absolute`: ``\langle C, YY^T\rangle - d^*``.
- `gtol`: Stationarity (gradient norm) tolerance. Interpretation depends on
    `gtol_mode`. The default value is ``0.0`` (not used as a stopping criterion).
- `gtol_mode`: Controls how the gradient norm is measured.
    * `:relative` (default): ``\|\nabla_Y \mathcal{L}\|_2 / \|C\|_2``.
    * `:absolute`: ``\|\nabla_Y \mathcal{L}\|_2``.
- `numlbfgsvecs`: Number of L-BFGS vectors. The default value is ``4``.
- `fprec`: Break one major iteration if the relative change of the
    Lagrangian value is smaller than `fprec * eps()`. The default value
    is ``10^8``, which is for moderate-accuracy solutions (ptol = objtol = ``10^{-2}``).
- `prior_trace_bound`: A trace bound priorly known or estimated.
    For example, it is ``n`` for max cut. The default value is ``10^{18}``.
- `σfac`: Factor for increasing the smoothing factor ``\sigma``
   in the augmented Lagrangian. The default value is ``2.0``.
- `rankupd_tol`: Rank update tolerance. After primal infeasibility
    reaches `ptol` and `objtol` is not reached for `rankupd_tol`
    major iterations, the rank of the solution matrix ``Y`` is doubled.
    The default value is ``4``.
- `maxtime`: Maximum time in seconds for the optimization. The default
    value is ``3600.0``. There may be some postprocessing overhead so
    the program will not stop exactly at `maxtime`. If you want to
    achieve a hard time limit, use terminal tools.
- `printlevel`: Print level. The default value is ``1``.
- `printfreq`: How often to print in seconds. The default value is ``60.0``.
- `maxmajoriter`: Maximum number of major iterations. The default value
    is ``10^5``.
- `maxiter`: Maximum number of total iterations. The default value is ``10^7``.
- `dataset`: Dataset name for better tracking progress,
    especially when executed parallely. The default value is "".
- `eval_DIMACS_errs`: Whether to evaluate DIMACS errors. The default value
    is false.
- `init_func`: Optional custom initialization function. Called as
    `init_func(data, r, init_args...)` and must return `(Rt0, λ0)` where
    `Rt0` is `r×n` and `λ0` is length-`m`. Used for initial setup and when
    rank is doubled in `rank_update!`. If not provided, random init is used.
- `init_args`: Tuple of extra arguments passed to `init_func` after `(data, r)`.
    Default is `()`.
"""
function sdplr(
    C::AbstractMatrix{Tv},
    As::Vector,
    b::Vector{Tv},
    r::Ti;
    constraint_types::Union{Nothing,AbstractVector{Bool}}=nothing,
    config::BurerMonteiroConfig{Ti,Tv}=BurerMonteiroConfig{Ti,Tv}(),
    kwargs...,
) where {Ti<:Integer,Tv}

    # update config with input kwargs
    for (key, value) in kwargs
        if hasfield(BurerMonteiroConfig, Symbol(key))
            setfield!(config, Symbol(key), value)
        else
            @error "Unrecognized keyword argument $key"
        end
    end

    if config.printlevel > 0
        printheading(1)
    end

    preprocess_dt = @elapsed begin
        data = if constraint_types === nothing
            SDPData(C, As, b)
        else
            SDPData(C, As, b, constraint_types)
        end
        var = SolverVars(data, r, config)
        aux = SolverAuxiliary(data)
        stats = SolverStats{Tv}()
    end

    @debug "preprocess dt" preprocess_dt

    ans = _sdplr(data, var, aux, stats, config)

    # record preprocessing time
    ans["preprocess_time"] = preprocess_dt
    ans["totaltime"] += preprocess_dt

    if config.printlevel > 0
        printheading(0)
    end

    return ans
end

"""
    sdplr(C, As, b, r)  [complex Hermitian variant]

Solve a complex Hermitian SDP by embedding it into an equivalent real symmetric
SDP of doubled dimension (2n×2n) and calling the real solver.

The embedding is:

    phi(A) = [ Re(A)   -Im(A) ]   (2n×2n real symmetric)
              [ Im(A)    Re(A) ]

with tr(phi(A)/2 * phi(X)) = Re(tr(A*X)), so the inner products are preserved
exactly.  All kwargs are forwarded to the underlying real solver unchanged.

The returned Dict contains:
- "Rt"  — complex primal factor of shape (r_final, n), where X ≈ Rt' * Rt
- "Rt0" — initial complex factor (same shape)
- all other fields have the same meaning as for real SDPs

The starting rank `r` is internally doubled to `2r` to match the degrees of
freedom of a complex rank-r factor.
"""
function sdplr(
    C::AbstractMatrix{Tv},
    As::Vector,
    b::AbstractVector,
    r::Ti;
    constraint_types::Union{Nothing,AbstractVector{Bool}}=nothing,
    kwargs...,
) where {Ti<:Integer,Tv<:Complex}
    n_orig = size(C, 1)
    C_real, As_real, b_real = embed_hermitian_sdp(C, As, b)
    result = sdplr(
        C_real, As_real, b_real, Ti(2 * r);
        constraint_types=constraint_types,
        kwargs...,
    )
    extract_complex_result!(result, n_orig)
    return result
end

function _sdplr(
    data,
    var::SolverVars{Ti,Tv},
    aux,
    stats::SolverStats{Tv},
    config::BurerMonteiroConfig{Ti,Tv},
) where {Ti<:Integer,Tv}
    n = side_dimension(aux)
    m = length(var.λ) # number of constraints

    stats.starttime[] = time()

    lastprint = stats.starttime[] # timestamp of last print

    Rt0 = deepcopy(var.Rt)
    Rt0 = deepcopy(var.Rt)
    λ0 = deepcopy(var.λ)

    # set up algorithm parameters
    normb = norm(b_vector(data), 2)
    normC = norm(C_matrix(data), 2)

    # initialize lbfgs datastructures
    lbfgshis = lbfgs_init(var.Rt, config.numlbfgsvecs)

    cur_gtol = 1.0 / var.σ[]     # stationarity tolerance
    cur_ptol = 1.0 / var.σ[]^0.1   # primal violation tolerance

    cur_ptol = max(cur_ptol, config.ptol)
    cur_gtol = max(cur_gtol, config.gtol)
    𝓛_val, grad_norm, primal_vio_norm = fg!(data, var, aux, normC, normb, config)

    iter = 0 # total number of iterations

    dirt = similar(var.Rt) # t means transpose
    majoriter = 0
    use_armijo = data.has_inequalities  # branch once; avoids scanning constraint_types per step

    rankupd_tol_cnt = config.rankupd_tol

    duality_gap = 1e20
    min_duality_gap = 1e20
    max_dual_value = -1e20
    best_λ = deepcopy(var.λ)

    for _ in 1:config.maxmajoriter
        majoriter += 1
        localiter = 0

        # find a stationary point of the Lagrangian
        while grad_norm > cur_gtol
            # update iteration counters
            localiter += 1
            iter += 1
            # find the lbfgs direction
            # the return direction has been negated
            lbfgs_dir_dt = @elapsed begin
                lbfgs_dir!(dirt, lbfgshis, var.Gt; negate=true)
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
                if use_armijo
                    α, 𝓛_val = linesearch_armijo!(var, aux, dirt; α_max=1.0)
                else
                    α, 𝓛_val = linesearch!(var, aux, dirt; α_max=1.0)
                end
            end
            @debug "line search time" linesearch_dt

            # update R and update gradient, stationarity, primal violence
            axpy!(α, dirt, var.Rt)
            g_dt = @elapsed begin
                g!(var, aux)
            end
            @debug "g time" g_dt
            grad_norm = if config.gtol_mode == :relative
                norm(var.Gt, 2) / normC
            else
                norm(var.Gt, 2)
            end

            primal_vio_norm = if config.ptol_mode == :relative
                norm(var.primal_vio, 2) / normb
            else
                norm(var.primal_vio, 2)
            end

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
            # if print frequency is reached, print intermediate results
            if current_time - lastprint >= config.printfreq
                lastprint = current_time
                if config.printlevel > 0
                    printintermediate(
                        config.dataset,
                        majoriter,
                        localiter,
                        iter,
                        𝓛_val,
                        var.obj[],
                        var.σ[],
                        cur_gtol,
                        cur_ptol,
                        grad_norm,
                        primal_vio_norm,
                        min_duality_gap,
                        max_dual_value,
                    )
                end
            end

            # timeout or iteration limit reached
            if (
                current_time - stats.starttime[] > config.maxtime ||
                iter > config.maxiter
            )
                break
            end
        end

        current_time = time()
        printintermediate(
            config.dataset,
            majoriter,
            localiter,
            iter,
            𝓛_val,
            var.obj[],
            var.σ[],
            cur_gtol,
            cur_ptol,
            grad_norm,
            primal_vio_norm,
            min_duality_gap,
            max_dual_value,
        )
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
            # when highprecision=true, then GenericArpack will be used
            # otherwise Lanczos with random start will be used
            dual_dt = @elapsed begin
                dual_value, _ = dual_obj(
                    data,
                    var,
                    aux,
                    config.prior_trace_bound,
                    iter;
                    highprecision=config.eigval_highprecision,
                )
            end

            if dual_value > max_dual_value
                best_λ = -deepcopy(var.y)
                max_dual_value = dual_value
            end
            duality_gap = if config.objtol_mode == :relative
                (var.obj[] - max_dual_value) / minimum(abs.([var.obj[], max_dual_value]))
            else
                var.obj[] - max_dual_value
            end
            stats.dual_time[] += dual_dt
            @show var.obj[] max_dual_value duality_gap
            if primal_vio_norm <= config.ptol
                @debug "primal vio is small enough, checking duality bound."
                if config.objtol == Inf
                    @debug "`objtol` is `Inf`, skipping duality gap check"
                    break
                end
                if duality_gap <= config.objtol
                    @debug "Duality gap and primal violence are small enough."
                    @debug primal_vio_norm duality_gap grad_norm
                    min_duality_gap = min(min_duality_gap, duality_gap)
                    break
                else
                    if min_duality_gap - duality_gap < config.objtol
                        rankupd_tol_cnt -= 1
                    else
                        rankupd_tol_cnt = config.rankupd_tol
                    end
                    min_duality_gap = min(min_duality_gap, duality_gap)
                    if rankupd_tol_cnt == 0
                        rank_double = true
                    end
                end
            end
            @inbounds for i in 1:m
                var.λ[i] = min(
                    var.λ_ub[i], var.λ[i] - var.σ[] * var.primal_vio_raw[i]
                )
            end
            cur_ptol = cur_ptol / var.σ[]^0.9
            cur_gtol = cur_gtol / var.σ[]
        else
            var.σ[] *= config.σfac
            cur_ptol = 1 / var.σ[]^0.1
            cur_gtol = 1 / var.σ[]
            cur_gtol = 1 / var.σ[]
        end

        # when objective gap doesn't improve, we double the rank
        if rank_double
            var = rank_update!(data, var, config)
            cur_ptol = 1 / var.σ[]^0.1
            cur_gtol = 1 / var.σ[]
            lbfgshis = lbfgs_init(var.Rt, config.numlbfgsvecs)
            dirt = similar(var.Rt)
            min_duality_gap = 1e20
            max_dual_value = -1e20
            rankupd_tol_cnt = config.rankupd_tol
            @info "rank doubled, newrank is $(var.r[])."
        else
            lbfgs_clear!(lbfgshis)
        end

        cur_ptol = max(cur_ptol, config.ptol)
        cur_gtol = max(cur_gtol, config.gtol)
        𝓛_val, grad_norm, primal_vio_norm = fg!(data, var, aux, normC, normb, config)

        if majoriter == config.maxmajoriter
            @warn "Major iteration limit exceeded. Stop optimizing."
        end
    end

    𝓛_val, grad_norm, primal_vio_norm = fg!(data, var, aux, normC, normb, config)

    printintermediate(
        config.dataset,
        majoriter,
        -1,
        iter,
        𝓛_val,
        var.obj[],
        var.σ[],
        cur_gtol,
        cur_ptol,
        grad_norm,
        primal_vio_norm,
        min_duality_gap,
        max_dual_value,
    )

    stats.endtime[] = time()

    totaltime = stats.endtime[] - stats.starttime[]

    stats.primal_time[] = (totaltime - stats.dual_time[])
    stats.DIMACS_time[] = @elapsed begin
        if config.eval_DIMACS_errs
            DIMACS_errs = DIMACS_errors(data, var, aux)
        else
            DIMACS_errs = zeros(6)
        end
    end
    return Dict([
        "Rt" => var.Rt,
        "lambda" => best_λ,
        "Rt0" => Rt0,
        "lambda0" => λ0,
        "sigma" => var.σ[],
        "grad_norm" => grad_norm,
        "primal_vio" => primal_vio_norm,
        "obj" => var.obj[],
        "max_dual_value" => max_dual_value,
        "min_duality_gap" => min_duality_gap,
        "totaltime" => totaltime,
        "dual_time" => stats.dual_time[],
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
