"""
    f!(data, aux, res)

Update the objective value, primal violence and compute 
the augmented Lagrangian value, 
    ℒ(R, λ, σ) = Tr(C RRᵀ) - λᵀ(𝒜(RRᵀ) - b) + σ/2 ||𝒜(RRᵀ) - b||^2
"""
function f!(
    data::SDPData{Ti, Tv, TC},
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
) where {Ti <: Integer, Tv, TC <: AbstractMatrix{Tv}}
    m = data.m # number of constraints

    # apply the operator 𝒜 to RRᵀ and compute the objective value
    𝒜!(aux.primal_vio, aux.UVt, aux, var.Rt)
    var.obj[] = aux.primal_vio[m+1]

    v = @view aux.primal_vio[1:m]
    @. v -= data.b
    return (var.obj[] - dot(var.λ, v) + var.σ[] * dot(v, v) / 2) 
end


"""
Compute 𝒜(UUᵀ) 
"""
function 𝒜!(
    𝒜_UUt::Vector{Tv},
    UUt::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
    Ut::Matrix{Tv},
) where {Ti <: Integer, Tv}
    fill!(𝒜_UUt, zero(Tv))
    # deal with sparse and diagonal constraints first
    # store results of 𝒜(UUᵀ)
    if aux.n_sparse_matrices > 0
        𝒜_sparse!(𝒜_UUt, UUt, aux, Ut)
    end
    # then deal with low-rank matrices
    if aux.n_symlowrank_matrices > 0
        𝒜_symlowrank!(𝒜_UUt, aux, Ut)
    end
end

"""
Compute 𝒜((UVᵀ + VUᵀ)/2)
"""
function 𝒜!(
    𝒜_UVt::Vector{Tv},
    UVt::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
    Ut::Matrix{Tv},
    Vt::Matrix{Tv};
) where {Ti <: Integer, Tv}
    fill!(𝒜_UVt, zero(Tv))
    # deal with sparse and diagonal constraints first
    # store results of 𝓐(UVᵀ + VUᵀ)/2
    if aux.n_sparse_matrices > 0
        𝒜_sparse!(𝒜_UVt, UVt, aux, Ut, Vt)
    end
    # then deal with low-rank matrices
    if aux.n_symlowrank_matrices > 0
        𝒜_symlowrank!(𝒜_UVt, aux, Ut, Vt)
    end
end


function 𝒜_sparse!(
    𝒜_UUt::Vector{Tv},
    UUt::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
    Ut::Matrix{Tv},
) where {Ti <: Integer, Tv}
    𝒜_sparse_formUUt!(UUt, aux.triu_sparse_S, Ut)

    m = length(aux.triu_sparse_S.nzval)
    n = aux.n_sparse_matrices
    C = SparseMatrixCSC(m, n, 
                        aux.triu_agg_sparse_A_matptr, 
                        aux.triu_agg_sparse_A_nzind,
                        aux.triu_agg_sparse_A_nzval_two)
    v = UUt' * C
    𝒜_UUt[aux.sparse_As_global_inds] .= @view(v[1, :])
end


function 𝒜_sparse!(
    𝒜_UVt::Vector{Tv},
    UVt::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
    Ut::Matrix{Tv},
    Vt::Matrix{Tv},
) where {Ti <: Integer, Tv}
    𝒜_sparse_formUVt!(UVt, aux.triu_sparse_S, Ut, Vt)

    m = length(aux.triu_sparse_S.nzval)
    n = aux.n_sparse_matrices
    C = SparseMatrixCSC(m, n, 
                        aux.triu_agg_sparse_A_matptr, 
                        aux.triu_agg_sparse_A_nzind,
                        aux.triu_agg_sparse_A_nzval_two)
    v = UVt' * C
    𝒜_UVt[aux.sparse_As_global_inds] .= @view(v[1, :])
end


function tr_UtAU(A::SymLowRankMatrix{Tv}, Ut::Matrix{Tv}) where {Tv}
    UtB = Ut * A.B
    @. UtB = UtB^2
    rmul!(UtB, A.D)
    return sum(UtB)
end

function tr_UtAV(
    A::SymLowRankMatrix{Tv}, 
    Ut::Matrix{Tv}, 
    Vt::Matrix{Tv}
) where {Tv}
    UtB = Ut * A.B
    VtB = Vt * A.B
    @. UtB *= VtB
    rmul!(UtB, A.D)
    return sum(UtB)
end

function 𝒜_symlowrank!(
    𝒜_UUt::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
    Ut::Matrix{Tv},
) where {Ti <: Integer, Tv}
    for i = 1:aux.n_symlowrank_matrices
        global_id = aux.symlowrank_As_global_inds[i]
        𝒜_UUt[global_id] = tr_UtAU(aux.symlowrank_As[i], Ut)
    end
end

function 𝒜_symlowrank!(
    𝒜_UVt::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
    Ut::Matrix{Tv},
    Vt::Matrix{Tv},
) where {Ti <: Integer, Tv}
    for i = 1:aux.n_symlowrank_matrices
        global_id = aux.symlowrank_As_global_inds[i]
        𝒜_UVt[global_id] = tr_UtAV(aux.symlowrank_As[i], Ut, Vt)
    end
end

function mydot(Rt, row, col)
    m = size(Rt, 1)
    rval = zero(eltype(Rt))
    @simd for i in 1:m
      @inbounds rval += Rt[i, row] * Rt[i, col]
    end 
    return rval 
end

function mydot(Ut, Vt, row, col)
    m = size(Ut, 1)
    rval = zero(eltype(Ut))
    @simd for i in 1:m
      @inbounds rval += Ut[i, row] * Vt[i, col]
    end
    @simd for i in 1:m
      @inbounds rval += Vt[i, row] * Ut[i, col]
    end
    return rval / 2
end

function 𝒜_sparse_formUUt!(
    UUt::Vector{Tv},
    triu_Sparse_S::SparseMatrixCSC{Tv, Ti},
    Ut::Matrix{Tv},
)where {Ti <: Integer, Tv}
    fill!(UUt, zero(Tv))
    colptr = triu_Sparse_S.colptr
    rowval = triu_Sparse_S.rowval
    for col in axes(Ut, 2)
        for nzi in colptr[col]:(colptr[col+1]-1)
            row = rowval[nzi]
            UUt[nzi] = mydot(Ut, col, row) 
        end
    end
end

function 𝒜_sparse_formUVt!(
    UVt::Vector{Tv},
    triu_Sparse_S::SparseMatrixCSC{Tv, Ti},
    Ut::Matrix{Tv},
    Vt::Matrix{Tv},
) where {Ti <: Integer, Tv}
    fill!(UVt, zero(Tv))
    colptr = triu_Sparse_S.colptr
    rowval = triu_Sparse_S.rowval
    for col in axes(Ut, 2)
        for nzi in colptr[col]:(colptr[col+1]-1)
            row = rowval[nzi]
            UVt[nzi] = mydot(Ut, Vt, col, row) 
        end
    end
end


function 𝒜t_preprocess_sparse!(
    sparse_S::SparseMatrixCSC{Tv, Ti},
    triu_sparse_S_nzval::Vector{Tv},
    v::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
) where{Ti <: Integer, Tv}
    fill!(triu_sparse_S_nzval, zero(Tv))
    m = length(triu_sparse_S_nzval)
    n = length(v) 
    C = SparseMatrixCSC(m, n,
                        aux.triu_agg_sparse_A_matptr,
                        aux.triu_agg_sparse_A_nzind,
                        aux.triu_agg_sparse_A_nzval_one)
    mul!(triu_sparse_S_nzval, C, v)

    @inbounds @simd for i = 1:length(aux.agg_sparse_A_mappedto_triu)
        j = aux.agg_sparse_A_mappedto_triu[i]
        sparse_S.nzval[i] = triu_sparse_S_nzval[j]
    end
end


function copy2y_λ_sub_pvio!(
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
)where {Ti <: Integer, Tv}  
    m = length(aux.primal_vio)-1
    @inbounds @simd for i = 1:m
        aux.y[i] = -(var.λ[i] - var.σ[] * aux.primal_vio[i])
    end 
    aux.y[m+1] = one(Tv)
end


function copy2y_λ!(
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
)where {Ti <: Integer, Tv}
    m = length(aux.primal_vio)-1
    @inbounds @simd for i = 1:m
        aux.y[i] = -var.λ[i]
    end 
    aux.y[m+1] = one(Tv)
end


function 𝒜t_preprocess!(
    aux::SolverAuxiliary{Ti, Tv},
) where {Ti <: Integer, Tv}
    # update the sparse matrix
    # compute C - ∑_{i=1}^{m} λ_i A_i for sparse matrices

    if aux.n_sparse_matrices > 0
        v = aux.y[aux.sparse_As_global_inds]
        𝒜t_preprocess_sparse!(aux.sparse_S, aux.triu_sparse_S.nzval, v, aux)
    end
end


function 𝒜t!(
    y::Tx, 
    x::Tx, 
    aux::SolverAuxiliary{Ti, Tv}, 
)where{Ti <: Integer, Tv, Tx <: AbstractArray{Tv}}
    # zero out output vector
    fill!(y, zero(Tv))

    # deal with sparse and diagonal constraints first
    if aux.n_sparse_matrices > 0
        mul!(y, x, aux.sparse_S)
    end

    # then deal with low-rank matrices 
    if aux.n_symlowrank_matrices > 0
        for i = 1:aux.n_symlowrank_matrices
            global_id = aux.symlowrank_As_global_inds[i]
            coeff = aux.y[global_id]
            mul!(y, x, aux.symlowrank_As[i], coeff, one(Tv))
        end 
    end
end


function 𝒜t!(
    y::Tx, 
    aux::SolverAuxiliary{Ti, Tv}, 
    x::Tx,
) where{Ti <: Integer, Tv, Tx <: AbstractArray{Tv}}
    # zero out output vector
    fill!(y, zero(Tv))

    # deal with sparse and diagonal constraints first
    if aux.n_sparse_matrices > 0
        mul!(y, aux.sparse_S, x)
    end

    # then deal with low-rank matrices 
    if aux.n_symlowrank_matrices > 0
        for i = 1:aux.n_symlowrank_matrices
            global_id = aux.symlowrank_As_global_inds[i]
            coeff = aux.y[global_id]
            mul!(y, aux.symlowrank_As[i], x, coeff, one(Tv))
        end 
    end
end


"""
This function computes the gradient of the augmented Lagrangian
"""
function g!(
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
) where{Ti <: Integer, Tv}
    𝒜t_preprocess_dt = @elapsed begin
        copy2y_λ_sub_pvio!(var, aux)
        𝒜t_preprocess!(aux)
    end
    densesparse_dt = @elapsed begin
        𝒜t!(var.Gt, var.Rt, aux)
    end
    @debug ("𝒜t_preprocess_dt: $𝒜t_preprocess_dt, 
        densesparse_dt: $densesparse_dt")
    BLAS.scal!(Tv(2), var.Gt)
    return 0
end


"""
Function for computing Lagrangian value, stationary condition 
and primal feasibility.
"""
function fg!(
    data::SDPData{Ti, Tv, TC},
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
    normC::Tv,
    normb::Tv,
) where {Ti <: Integer, Tv, TC <: AbstractMatrix{Tv}}
    m = data.m
    f_dt = @elapsed begin
        𝓛_val = f!(data, var, aux)
    end
    g_dt = @elapsed begin
        g!(var, aux)
    end
    @debug "f dt, g dt" f_dt, g_dt
    grad_norm = norm(var.Gt, 2) / (1.0 + normC)

    v = @view aux.primal_vio[1:m]
    primal_vio_norm = norm(v, 2) / (1.0 + normb)
    return (𝓛_val, grad_norm, primal_vio_norm)
end


function SDP_S_eigval(
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
    nevs::Ti,
    preprocessed::Bool=false;
    kwargs...
) where {Ti <: Integer, Tv}
    if !preprocessed
        𝒜t_preprocess!(aux)
    end
    n = size(aux.sparse_S, 1)
    GenericArpack_dt = @elapsed begin
        op = ArpackSimpleFunctionOp(
            (y, x) -> begin
                𝒜t!(y, aux, x)
                # shift the matrix by I
                y .+= x
                return y 
            end, n)
        GenericArpack_eigvals, _ = symeigs(op, nevs; kwargs...)
    end
    GenericArpack_eigvals = real.(GenericArpack_eigvals)
    GenericArpack_eigvals .-= 1 # cancel the shift
    return GenericArpack_eigvals, GenericArpack_dt
end


function surrogate_duality_gap(
    data::SDPData{Ti, Tv, TC},
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
    trace_bound::Tv, 
    iter::Ti;
    highprecision::Bool=false,
) where {Ti <: Integer, Tv, TC <: AbstractMatrix{Tv}}
    copy2y_λ_sub_pvio!(var, aux)

    𝒜t_preprocess!(aux)
    lanczos_dt = @elapsed begin
        lanczos_eigenval = approx_mineigval_lanczos(aux, iter)
    end
    res = lanczos_eigenval
    if highprecision
        n = size(aux.sparse_S, 1)
        GenericArpack_evs, GenericArpack_dt = 
            SDP_S_eigval(var, aux, 1, true; which=:SA,
                         ncv=min(100, n), tol=1e-6, maxiter=1000000)
        res = GenericArpack_evs[1]
    else
        GenericArpack_dt = 0.0
        GenericArpack_evs = [0.0]
    end

    m = length(data.b)
    duality_gap = (var.obj[] + dot(aux.y[1:m], data.b) - 
             trace_bound * min(res[1], 0.0))     
    rel_duality_gap = duality_gap / max(one(Tv), abs(var.obj[])) 

    return lanczos_dt, lanczos_eigenval, GenericArpack_dt, 
           res[1], duality_gap, rel_duality_gap
end


"""
Function for computing six DIMACS_errors
    error1 = ||𝒜(X) - b||₂ / (1 + ||b||₂)
    error2 = max {-λ_min(X), 0} / (1 + ||b||₂)
    error3 = ||𝒜^*(y) + Z - C||_F / (1 + ||C||_F)
    error4 = max {-λ_min(Z), 0} / (1 + ||C||_F) 
    error5 = (<C, X> - b^T y) / (1 + |<C, X>| + |b^T y|)  
    error6 = <X, Z> / (1 + |<C, X>| + |b^T y|)
"""
function DIMACS_errors(
    data::SDPData{Ti, Tv, TC},
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
) where {Ti <: Integer, Tv, TC <: AbstractMatrix{Tv}}
    n = size(aux.sparse_S, 1)
    v = @view aux.primal_vio[1:data.m] 
    err1 = norm(v, 2) / (1.0 + norm(data.b, 2))       
    err2 = 0.0
    err3 = 0.0 # err2, err3 are zero as X = YY^T, Z = C - 𝒜^*(y)

    copy2y_λ!(var, aux)
    𝒜t_preprocess!(aux)

    GenericArpack_evs, _ = 
        SDP_S_eigval(var, aux, 1, true; 
                    which=:SA, ncv=min(100, n), maxiter=1000000)
    err4 = (max(zero(Tv), -real.(GenericArpack_evs[1])) 
            / (1.0 + norm(data.C, 2)))
    err5 = ((var.obj[] - dot(var.λ, data.b)) / 
            (1.0 + abs(var.obj[]) + abs(dot(var.λ, data.b))))
    err6 = (dot(var.Rt, var.Rt * aux.sparse_S) / 
            (1.0 + abs(var.obj[]) + abs(dot(var.λ, data.b))))
    return [err1, err2, err3, err4, err5, err6]
end


"""
Approximate the minimum eigenvalue of a symmetric matrix `A`.

Perform `q` Lanczos iterations with *a random start vector* to approximate 
the minimum eigenvalue of `A`.
"""
function approx_mineigval_lanczos(
    aux::SolverAuxiliary{Ti, Tv},
    q::Ti,
) where {Ti <: Integer, Tv}
    n::Ti = size(aux.sparse_S, 1)
    q = min(q, n - 1)

    # allocate lanczos vectors
    # alpha is the diagonal of the tridiagonal matrix
    # beta is the subdiagonal of the tridiagonal matrix
    alpha = zeros(q, 1)
    beta = zeros(q, 1)

    v = randn(Tv, n)
    v ./= norm(v, 2)

    Av = zeros(Tv, n)
    v_pre = zeros(Tv, n)
    
    iter = 0

    Btvs = Vector{Tv}[]
    for _ = 1:aux.n_symlowrank_matrices
        s = size(aux.symlowrank_As[1].B, 2)
        push!(Btvs, zeros(Tv, s))
    end

    for i = 1:q
        iter += 1
        𝒜t!(Av, aux, v)
        alpha[i] = v' * Av

        if i == 1
            @. Av -= alpha[i] * v
        else
            @. Av -= alpha[i] * v + beta[i-1] * v_pre
        end

        beta[i] = norm(Av, 2)

        if  abs(beta[i]) < sqrt(n) * eps() 
            break
        end
        Av ./= beta[i]
        copyto!(v_pre, v)
        copyto!(v, Av)
    end
    # shift the matrix by I
    alpha .+= 1
    B = SymTridiagonal(alpha[1:iter], beta[1:iter-1])
    @info "Symmetric tridiagonal matrix formed."
    min_eigval, _ = 
        symeigs(B, 1; which=:SA, ncv=minimum([100, q, n]), maxiter=1000000, tol=1e-4)
    return real.(min_eigval)[1] - 1 # cancel the shift
end


function rank_update!(
    var::SolverVars{Ti, Tv},
) where {Ti <: Integer, Tv}
    n = size(var.Rt, 2)
    m = size(var.λ, 1)
    r = var.r[]
    max_r = barvinok_pataki(n, m)
    newr = min(max_r, r * 2)

    newRt = 2 * rand(Tv, newr, n) .- 1
    newλ = randn(m)

    return SolverVars(
        newRt, 
        zeros(Tv, size(newRt)), 
        newλ, 
        Ref(Ti(newr)),
        Ref(Tv(2.0)),
        Ref(Tv(0.0))
    )
end