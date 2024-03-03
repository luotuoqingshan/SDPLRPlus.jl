"""
    f!(SDP)

Update the objective value, primal violence and compute 
the augmented Lagrangian value, 
    𝓛(R, λ, σ) = Tr(C RRᵀ) - λᵀ(𝓐(RRᵀ) - b) + σ/2 ||𝓐(RRᵀ) - b||^2
"""
function f!(SDP::SDPProblem{Ti, Tv, TC}
    ) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    # apply the operator 𝓐 to RRᵀ and compute the objective value
    SDP.obj = 𝒜!(SDP.primal_vio, SDP.UVt, SDP, SDP.R, SDP.R; same=true)
    SDP.primal_vio .-= SDP.b 
    return (SDP.obj - dot(SDP.λ, SDP.primal_vio)
           + SDP.σ * dot(SDP.primal_vio, SDP.primal_vio) / 2) 
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
    SDP::SDPProblem{Ti, Tv, TC},
    U::Matrix{Tv},
    V::Matrix{Tv};
    same::Bool=true,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    fill!(𝓐_UV, zero(eltype(𝓐_UV)))
    obj = zero(Tv) 
    # deal with sparse and diagonal constraints first
    # store results of 𝓐(UVᵀ + VUᵀ)/2
    if SDP.n_sparse_matrices > 0
        Aoper_formUVt!(UVt, SDP, U, V; same=same) 
        for i = 1:SDP.n_sparse_matrices
            res = zero(Tv) 
            for j = SDP.agg_A_ptr[i]:(SDP.agg_A_ptr[i + 1] - 1)
                res += SDP.agg_A_nzval_two[j] * UVt[SDP.agg_A_nzind[j]]
            end
            if SDP.sparse_As_global_inds[i] == 0
                obj = res
            else
                𝓐_UV[SDP.sparse_As_global_inds[i]] = res
            end
        end
    end
    # then deal with low-rank matrices
    if SDP.n_symlowrank_matrices > 0
        if same
            for i = 1:SDP.n_symlowrank_matrices
                mul!(SDP.BtUs[i], SDP.symlowrank_As[i].Bt, U)
                @. SDP.BtUs[i] = SDP.BtUs[i]^2
                lmul!(SDP.symlowrank_As[i].D, SDP.BtUs[i])
                res = sum(SDP.BtUs[i])

                if SDP.symlowrank_As_global_inds[i] == 0
                    obj = res
                else
                    𝓐_UV[SDP.symlowrank_As_global_inds[i]] = res 
                end
            end
        else
            for i = 1:SDP.n_symlowrank_matrices
                mul!(SDP.BtUs[i], SDP.symlowrank_As[i].Bt, U)
                mul!(SDP.BtVs[i], SDP.symlowrank_As[i].Bt, V)
                @. SDP.BtUs[i] *= SDP.BtVs[i]
                lmul!(SDP.symlowrank_As[i].D, SDP.BtUs[i])
                res = sum(SDP.BtUs[i])

                if SDP.symlowrank_As_global_inds[i] == 0
                    obj = res
                else
                    𝓐_UV[SDP.symlowrank_As_global_inds[i]] = res 
                end
            end
        end
    end
    return obj
end


function Aoper_formUVt!(
    UVt::Vector{Tv},
    SDP::SDPProblem{Ti, Tv, TC},
    U::Matrix{Tv},
    V::Matrix{Tv};
    same::Bool=true,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    fill!(UVt, zero(eltype(UVt)))
    Ut = U' 
    if same
        @inbounds @simd for col in 1:SDP.n
            for nzind in SDP.XS_colptr[col]:(SDP.XS_colptr[col + 1] - 1)
                row = SDP.XS_rowval[nzind]
                UVt[nzind] = dot(@view(Ut[:, col]), @view(Ut[:, row]))
            end
        end
    else
        Vt = V'
        @inbounds @simd for col in 1:SDP.n
            for nzind in SDP.XS_colptr[col]:(SDP.XS_colptr[col + 1] - 1)
                row = SDP.XS_rowval[nzind]
                UVt[nzind] = dot(@view(Ut[:, col]), @view(Vt[:, row]))
                UVt[nzind] += dot(@view(Vt[:, col]), @view(Ut[:, row]))
                UVt[nzind] /= Tv(2)
            end
        end
    end
end


"""
    AToper!(o, triu_o_nzval, v, SDP)
"""
function AToper_preprocess_sparse!(
    o::SparseMatrixCSC{Tv, Ti},
    triu_o_nzval::Vector{Tv},
    v::Vector{Tv},
    SDP::SDPProblem{Ti, Tv, TC},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    fill!(triu_o_nzval, zero(Tv))
    for i = 1:SDP.n_sparse_matrices
        ind = SDP.sparse_As_global_inds[i]
        coeff = ind == 0 ? one(Tv) : v[ind]
        for j = SDP.agg_A_ptr[i]:(SDP.agg_A_ptr[i + 1] - 1)
            triu_o_nzval[SDP.agg_A_nzind[j]] += SDP.agg_A_nzval_one[j] * coeff
        end
    end

    @inbounds @simd for i = 1:length(SDP.full_S_triu_S_inds)
        o.nzval[i] = triu_o_nzval[SDP.full_S_triu_S_inds[i]]
    end
end


function AToper_preprocess!(
    SDP::SDPProblem{Ti, Tv, TC},
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    # update auxiliary vector y based on primal violence and λ
    # and then update the sparse matrix
    @. SDP.y = -(SDP.λ - SDP.σ * SDP.primal_vio)
    if SDP.n_sparse_matrices > 0
        AToper_preprocess_sparse!(SDP.full_S, SDP.S_nzval, SDP.y, SDP)
    end
end


function AToper!(
    y::Tx,
    SDP::SDPProblem{Ti, Tv, TC},
    Btvs::Vector{Tx},
    x::Tx, 
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, Tx <: AbstractArray{Tv}}
    # zero out output vector
    fill!(y, zero(Tv))

    # deal with sparse and diagonal constraints first
    if SDP.n_sparse_matrices > 0
        y .= SDP.full_S * x
    end

    # then deal with low-rank matrices 
    if SDP.n_symlowrank_matrices > 0
        for i = 1:SDP.n_symlowrank_matrices
            mul!(Btvs[i], SDP.symlowrank_As[i].Bt, x)
            lmul!(SDP.symlowrank_As[i].D, Btvs[i])
            coeff = SDP.symlowrank_As_global_inds[i] == 0 ? one(Tv) : SDP.y[SDP.symlowrank_As_global_inds[i]]
            mul!(y, SDP.symlowrank_As[i].B, Btvs[i], coeff, one(Tv))
        end 
    end
end


"""
This function computes the gradient of the augmented Lagrangian
"""
function g!(
    SDP::SDPProblem{Ti, Tv, TC},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    AToper_preprocess!(SDP)
    AToper!(SDP.G, SDP, SDP.BtUs, SDP.R)
    BLAS.scal!(Tv(2), SDP.G)
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
    SDP::SDPProblem{Ti, Tv, TC},
    normC::Tv,
    normb::Tv,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    f_dt = @elapsed begin
        𝓛_val = f!(SDP)
    end
    g_dt = @elapsed begin
        g!(SDP)
    end
    @debug "f dt, g dt" f_dt, g_dt
    grad_norm = norm(SDP.G, 2) / (1.0 + normC)
    primal_vio_norm = norm(SDP.primal_vio, 2) / (1.0 + normb)
    return (𝓛_val, grad_norm, primal_vio_norm)
end


"""

"""
function SDP_S_eigval(
    SDP::SDPProblem{Ti, Tv, TC},
    nevs::Ti,
    preprocessed::Bool=false;
    kwargs...
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    if !preprocessed
        AToper_preprocess!(SDP)
    end
    n = size(SDP.full_S, 1)
    GenericArpack_dt = @elapsed begin
        op = ArpackSimpleFunctionOp(
            (y, x) -> begin
                fill!(y, zero(Tv))
                if SDP.n_sparse_matrices > 0
                    y .= SDP.full_S * x 
                end
                if SDP.n_symlowrank_matrices > 0
                    for i = 1:SDP.n_symlowrank_matrices
                        coeff = SDP.symlowrank_As_global_inds[i] == 0 ? one(Tv) : SDP.y[SDP.symlowrank_As_global_inds[i]]
                        mul!(y, SDP.symlowrank_As[i], x, coeff, one(Tv))
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
    SDP::SDPProblem{Ti, Tv, TC}, 
    trace_bound::Tv, 
    iter::Ti;
    highprecision::Bool=false,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    AX = SDP.primal_vio + SDP.b
    AToper_preprocess!(SDP)
    n = size(SDP.full_S, 1)
    lanczos_dt = @elapsed begin
        lanczos_eigenval = approx_mineigval_lanczos(SDP, iter)
    end
    res = lanczos_eigenval
    if highprecision
        GenericArpack_evs, GenericArpack_dt = SDP_S_eigval(SDP, 1, true; which=:SA, tol=1e-6, maxiter=1000000)
        res = GenericArpack_evs[1]
    end

    duality_gap = (SDP.obj - dot(SDP.λ, SDP.b) + SDP.σ/2 * dot(SDP.primal_vio, AX + SDP.b)
           - max(trace_bound, norm(SDP.R)^2) * min(res[1], 0.0))     
    rel_duality_gap = duality_gap / max(one(Tv), abs(SDP.obj)) 
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
    SDP::SDPProblem{Ti, Tv, TC},
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    err1 = norm(SDP.primal_vio, 2) / (1.0 + norm(SDP.b, 2))       
    err2 = 0.0
    err3 = 0.0 # err2, err3 are zero as X = YY^T, Z = C - 𝒜^*(y)
    AToper_preprocess!(SDP)
    n = size(SDP.full_S, 1)

    GenericArpack_evs, GenericArpack_dt = SDP_S_eigval(SDP, 1, true; which=:SA, tol=1e-6, maxiter=1000000)
    err4 = max(zero(Tv), -real.(GenericArpack_evs[1])) / (1.0 + norm(SDP.C, 2))
    err5 = (SDP.obj - dot(SDP.λ, SDP.b)) / (1.0 + abs(SDP.obj) + abs(dot(SDP.λ, SDP.b)))
    err6 = dot(SDP.R, SDP.full_S, SDP.R) / (1.0 + abs(SDP.obj) + abs(dot(SDP.λ, SDP.b)))
    return [err1, err2, err3, err4, err5, err6]
end


"""
Approximate the minimum eigenvalue of a symmetric matrix `A`.

Perform `q` Lanczos iterations with *a random start vector* to approximate 
the minimum eigenvalue of `A`.
"""
function approx_mineigval_lanczos(
    SDP::SDPProblem{Ti, Tv, TC},
    q::Ti,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    n::Ti = size(SDP.full_S, 1)
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

    for i = 1:q
        iter += 1
        AToper!(Av, SDP, SDP.Btvs, v)
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