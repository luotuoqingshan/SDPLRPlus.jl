"""
    f!(data, aux, res)

Update the objective value, primal violence and compute 
the augmented Lagrangian value, 
    𝓛(R, λ, σ) = Tr(C RRᵀ) - λᵀ(𝓐(RRᵀ) - b) + σ/2 ||𝓐(RRᵀ) - b||^2
"""
function f!(
    data::SDPData{Ti, Tv, TC},
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    # apply the operator 𝓐 to RRᵀ and compute the objective value
    var.obj[] = 𝒜!(aux.primal_vio, aux.UVt, aux, var.R, var.R; same=true)
    aux.primal_vio .-= data.b 
    return (var.obj[] - dot(var.λ, aux.primal_vio)
           + var.σ[] * dot(aux.primal_vio, aux.primal_vio) / 2) 
end


"""
This function computes the violation of constraints,
i.e. it computes 𝓐((UVᵀ + VUᵀ)/2)

same : 1 if U and V are the same matrix
     : 0 if U and V are different matrices
obj  : whether to compute the objective function value
"""
function 𝒜!(
    𝓐_UV::Vector{Tv},
    UVt::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
    U::Matrix{Tv},
    V::Matrix{Tv};
    same::Bool=true,
) where {Ti <: Integer, Tv <: AbstractFloat}
    fill!(𝓐_UV, zero(eltype(𝓐_UV)))
    obj = zero(Tv) 
    # deal with sparse and diagonal constraints first
    # store results of 𝓐(UVᵀ + VUᵀ)/2
    if aux.n_sparse_matrices > 0
        Aoper_formUVt!(UVt, aux, U, V; same=same) 
        for i = 1:aux.n_sparse_matrices
            val = zero(Tv) 
            for j = aux.agg_sparse_A_matptr[i]:(aux.agg_sparse_A_matptr[i+1]-1)
                val += aux.agg_sparse_A_nzval_two[j] * UVt[aux.agg_sparse_A_nzind[j]]
            end
            if aux.sparse_As_global_inds[i] == 0
                obj = val
            else
                𝓐_UV[aux.sparse_As_global_inds[i]] = val
            end
        end
    end
    # then deal with low-rank matrices
    if aux.n_symlowrank_matrices > 0
        if same
            for i = 1:aux.n_symlowrank_matrices
                mul!(aux.BtUs[i], aux.symlowrank_As[i].Bt, U)
                @. aux.BtUs[i] = aux.BtUs[i]^2
                lmul!(aux.symlowrank_As[i].D, aux.BtUs[i])
                val = sum(aux.BtUs[i])

                if aux.symlowrank_As_global_inds[i] == 0
                    obj = val
                else
                    𝓐_UV[aux.symlowrank_As_global_inds[i]] = val 
                end
            end
        else
            for i = 1:aux.n_symlowrank_matrices
                mul!(aux.BtUs[i], aux.symlowrank_As[i].Bt, U)
                mul!(aux.BtVs[i], aux.symlowrank_As[i].Bt, V)
                @. aux.BtUs[i] *= aux.BtVs[i]
                lmul!(aux.symlowrank_As[i].D, aux.BtUs[i])
                val = sum(aux.BtUs[i])

                if aux.symlowrank_As_global_inds[i] == 0
                    obj = val
                else
                    𝓐_UV[aux.symlowrank_As_global_inds[i]] = val 
                end
            end
        end
    end
    return obj
end


function Aoper_formUVt!(
    UVt::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
    U::Matrix{Tv},
    V::Matrix{Tv};
    same::Bool=true,
) where {Ti <: Integer, Tv <: AbstractFloat}
    fill!(UVt, zero(eltype(UVt)))
    Ut = U' 
    n = size(U, 1)
    if same
        @inbounds @simd for col in 1:n
            for nzind in aux.triu_sparse_S.colptr[col]:(aux.triu_sparse_S.colptr[col+1]-1)
                row = aux.triu_sparse_S.rowval[nzind]
                UVt[nzind] = dot(@view(Ut[:, col]), @view(Ut[:, row]))
            end
        end
    else
        Vt = V'
        @inbounds @simd for col in 1:n
            for nzind in aux.triu_sparse_S.colptr[col]:(aux.triu_sparse_S.colptr[col+1]-1)
                row = aux.triu_sparse_S.rowval[nzind]
                UVt[nzind] = dot(@view(Ut[:, col]), @view(Vt[:, row]))
                UVt[nzind] += dot(@view(Vt[:, col]), @view(Ut[:, row]))
                UVt[nzind] /= Tv(2)
            end
        end
    end
end


function AToper_preprocess_sparse!(
    sparse_S::SparseMatrixCSC{Tv, Ti},
    triu_sparse_S_nzval::Vector{Tv},
    v::Vector{Tv},
    aux::SolverAuxiliary{Ti, Tv},
) where{Ti <: Integer, Tv <: AbstractFloat}
    fill!(triu_sparse_S_nzval, zero(Tv))
    for i = 1:aux.n_sparse_matrices
        ind = aux.sparse_As_global_inds[i]
        coeff = ind == 0 ? one(Tv) : v[ind]
        for j = aux.agg_sparse_A_matptr[i]:(aux.agg_sparse_A_matptr[i+1] - 1)
            triu_sparse_S_nzval[aux.agg_sparse_A_nzind[j]] += aux.agg_sparse_A_nzval_one[j] * coeff
        end
    end

    @inbounds @simd for i = 1:length(aux.agg_sparse_A_mappedto_triu)
        sparse_S.nzval[i] = triu_sparse_S_nzval[aux.agg_sparse_A_mappedto_triu[i]]
    end
end


function AToper_preprocess!(
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
) where {Ti <: Integer, Tv <: AbstractFloat}
    # update auxiliary vector y based on primal violence and λ
    # and then update the sparse matrix
    @. aux.y = -(var.λ - var.σ[] * aux.primal_vio)
    if aux.n_sparse_matrices > 0
        AToper_preprocess_sparse!(aux.sparse_S, aux.triu_sparse_S.nzval, aux.y, aux)
    end
end


function AToper!(
    y::Tx,
    aux::SolverAuxiliary{Ti, Tv},
    Btxs::Vector{Tx},
    x::Tx, 
) where{Ti <: Integer, Tv <: AbstractFloat, Tx <: AbstractArray{Tv}}
    # zero out output vector
    fill!(y, zero(Tv))

    # deal with sparse and diagonal constraints first
    if aux.n_sparse_matrices > 0
        y .= aux.sparse_S * x
    end

    # then deal with low-rank matrices 
    if aux.n_symlowrank_matrices > 0
        for i = 1:aux.n_symlowrank_matrices
            mul!(Btxs[i], aux.symlowrank_As[i].Bt, x)
            lmul!(aux.symlowrank_As[i].D, Btxs[i])
            coeff = aux.symlowrank_As_global_inds[i] == 0 ? one(Tv) : aux.y[aux.symlowrank_As_global_inds[i]]
            mul!(y, aux.symlowrank_As[i].B, Btxs[i], coeff, one(Tv))
        end 
    end
end


"""
This function computes the gradient of the augmented Lagrangian
"""
function g!(
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
) where{Ti <: Integer, Tv <: AbstractFloat}
    AToper_preprocess!(var, aux)
    AToper!(var.G, aux, aux.BtUs, var.R)
    BLAS.scal!(Tv(2), var.G)
    return 0
end


"""
Function for computing Lagrangian value, stationary condition and 
    primal feasibility
val : Lagrangian value
ρ_c_val : stationary condition
ρ_f_val : primal feasibility
"""
function fg!(
    data::SDPData{Ti, Tv, TC},
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
    normC::Tv,
    normb::Tv,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    f_dt = @elapsed begin
        𝓛_val = f!(data, var, aux)
    end
    g_dt = @elapsed begin
        g!(var, aux)
    end
    @debug "f dt, g dt" f_dt, g_dt
    grad_norm = norm(var.G, 2) / (1.0 + normC)
    primal_vio_norm = norm(aux.primal_vio, 2) / (1.0 + normb)
    return (𝓛_val, grad_norm, primal_vio_norm)
end


"""

"""
function SDP_S_eigval(
    var::SolverVars{Ti, Tv},
    aux::SolverAuxiliary{Ti, Tv},
    nevs::Ti,
    preprocessed::Bool=false;
    kwargs...
) where {Ti <: Integer, Tv <: AbstractFloat}
    if !preprocessed
        AToper_preprocess!(var, aux)
    end
    n = size(aux.sparse_S, 1)
    GenericArpack_dt = @elapsed begin
        op = ArpackSimpleFunctionOp(
            (y, x) -> begin
                fill!(y, zero(Tv))
                if aux.n_sparse_matrices > 0
                    y .= aux.sparse_S * x 
                end
                if aux.n_symlowrank_matrices > 0
                    for i = 1:aux.n_symlowrank_matrices
                        coeff = aux.symlowrank_As_global_inds[i] == 0 ? one(Tv) : aux.y[aux.symlowrank_As_global_inds[i]]
                        mul!(y, aux.symlowrank_As[i], x, coeff, one(Tv))
                    end 
                end
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
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    AX = aux.primal_vio + data.b
    AToper_preprocess!(var, aux)
    lanczos_dt = @elapsed begin
        lanczos_eigenval = approx_mineigval_lanczos(aux, iter)
    end
    res = lanczos_eigenval
    if highprecision
        GenericArpack_evs, GenericArpack_dt = SDP_S_eigval(var, aux, 1, true; which=:SA, tol=1e-6, maxiter=1000000)
        res = GenericArpack_evs[1]
    end

    duality_gap = (var.obj[] - dot(var.λ, data.b) + var.σ[]/2 * dot(aux.primal_vio, AX + data.b)
           - max(trace_bound, sum((var.R).^2)) * min(res[1], 0.0))     
    rel_duality_gap = duality_gap / max(one(Tv), abs(var.obj[])) 
    return lanczos_dt, lanczos_eigenval, GenericArpack_dt, GenericArpack_evs[1], duality_gap, rel_duality_gap 
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
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    err1 = norm(aux.primal_vio, 2) / (1.0 + norm(data.b, 2))       
    err2 = 0.0
    err3 = 0.0 # err2, err3 are zero as X = YY^T, Z = C - 𝒜^*(y)
    AToper_preprocess!(var, aux)

    GenericArpack_evs, _ = SDP_S_eigval(var, aux, 1, true; which=:SA, tol=1e-6, maxiter=1000000)
    err4 = max(zero(Tv), -real.(GenericArpack_evs[1])) / (1.0 + norm(data.C, 2))
    err5 = (var.obj[] - dot(var.λ, data.b)) / (1.0 + abs(var.obj[]) + abs(dot(var.λ, data.b)))
    err6 = dot(var.R, aux.sparse_S, var.R) / (1.0 + abs(var.obj[]) + abs(dot(var.λ, data.b)))
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
) where {Ti <: Integer, Tv <: AbstractFloat}
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
        AToper!(Av, aux, Btvs, v)
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
    B = SymTridiagonal(alpha[1:iter], beta[1:iter-1])
    min_eigval, _ = symeigs(B, 1; which=:SA, maxiter=1000000, tol=1e-6)
    return real.(min_eigval)[1]
end