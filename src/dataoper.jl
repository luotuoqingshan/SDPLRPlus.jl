"""
This function computes the augmented Lagrangian value, 
    𝓛(R, λ, σ) = Tr(C RRᵀ) - λᵀ(𝓐(RRᵀ) - b) + σ/2 ||𝓐(RRᵀ) - b||^2
"""
function lagrangval!(
    SDP::SDPProblem{Ti, Tv, TC}, 
    ) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    # apply the operator 𝓐 to RRᵀ and 
    # potentially compute the objective function value
    SDP.scalars.obj = Aoper!(SDP.primal_vio, SDP.UVt, SDP, SDP.R, SDP.R; same=true)
    SDP.primal_vio .-= SDP.b 
    return (SDP.scalars.obj - dot(SDP.λ, SDP.primal_vio)
           + SDP.scalars.σ * dot(SDP.primal_vio, SDP.primal_vio) / 2) 
end


"""
This function computes the violation of constraints,
i.e. it computes 𝓐((UVᵀ + VUᵀ)/2)

same : 1 if U and V are the same matrix
     : 0 if U and V are different matrices
obj  : whether to compute the objective function value
"""
function Aoper!(
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
    Aoper_formUVt!(UVt, SDP, U, V; same=same) 
    for i = 1:SDP.n_spase_matrices
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


function AToper!(
    o::SparseMatrixCSC{Tv, Ti},
    triu_o_nzval::Vector{Tv},
    v::Vector{Tv},
    SDP::SDPProblem{Ti, Tv, TC},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    fill!(triu_o_nzval, zero(Tv))
    for i = 1:SDP.n_spase_matrices
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

"""
This function computes the gradient of the augmented Lagrangian
"""
function gradient!(
    SDP::SDPProblem{Ti, Tv, TC},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    @. SDP.y = -(SDP.λ - SDP.scalars.σ * SDP.primal_vio)

    AToper!(SDP.full_S, SDP.S_nzval, SDP.y, SDP)

    #fill!(SDP.S_nzval, zero(Tv))
    #for i = 1:SDP.n_spase_matrices
    #    ind = SDP.sparse_As_global_inds[i]
    #    coeff = ind == 0 ? one(Tv) : SDP.y[ind]
    #    for j = SDP.agg_A_ptr[i]:(SDP.agg_A_ptr[i + 1] - 1)
    #        SDP.S_nzval[SDP.agg_A_nzind[j]] += SDP.agg_A_nzval_one[j] * coeff
    #    end
    #end

    #@inbounds @simd for i = 1:length(SDP.full_S_triu_S_inds)
    #    SDP.full_S.nzval[i] = SDP.S_nzval[SDP.full_S_triu_S_inds[i]]
    #end

    fill!(SDP.G, zero(Tv))
    SDP.G .= SDP.full_S * SDP.R 
    LinearAlgebra.BLAS.scal!(Tv(2), SDP.G)
    return 0
end


"""
Function for computing Lagrangian value, stationary condition and 
    primal feasibility
val : Lagrangian value
ρ_c_val : stationary condition
ρ_f_val : primal feasibility
"""
function essential_calcs!(
    SDP::SDPProblem{Ti, Tv, TC},
    normC::Tv,
    normb::Tv,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    L_val_dt = @elapsed begin
        𝓛_val = lagrangval!(SDP)
    end
    grad_dt = @elapsed begin
        gradient!(SDP)
    end
    stationarity = norm(SDP.G, 2) / (1.0 + normC)
    primal_vio = norm(SDP.primal_vio, 2) / (1.0 + normb)
    #@show L_val_dt, grad_dt
    return (𝓛_val, stationarity, primal_vio)
end


function surrogate_duality_gap(
    SDP::SDPProblem{Ti, Tv, TC}, 
    trace_bound::Tv, 
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    AX = SDP.primal_vio + SDP.b
    AToper!(SDP.full_S, SDP.S_nzval, -SDP.λ + SDP.scalars.σ * SDP.primal_vio, SDP)
    n = size(SDP.full_S, 1)
    eigval_dt1 = @elapsed begin
        eigenvals, _ = symeigs(SDP.full_S, 1; which=:SA, tol=1e-6, maxiter=1000000)
        @show real.(eigenvals[1])
    end
    eigval_dt2 = @elapsed begin
        eigenval = approx_mineigval_lanczos(SDP.full_S, 100)
        @show eigenval
    end
    eigval_dt3 = @elapsed begin
        decomp, history = partialschur(SDP.full_S, which=SR(), tol=1e-6)
        λs, X = partialeigen(decomp)
        @show λs
    end
    @show eigval_dt1, eigval_dt2, eigval_dt3
    duality_gap = (SDP.scalars.obj - dot(SDP.λ, SDP.b) + SDP.scalars.σ/2 * dot(SDP.primal_vio, AX + SDP.b)
           - max(trace_bound, norm(SDP.R)^2) * real.(eigenvals[1]))     
    rel_duality_gap = duality_gap / max(one(Tv), abs(SDP.scalars.obj)) 
    return duality_gap, rel_duality_gap 
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
    AToper!(SDP.full_S, SDP.S_nzval, -SDP.λ, SDP)
    n = size(SDP.full_S, 1)
    op = ArpackSimpleFunctionOp(
        (y, x) -> begin
                LinearAlgebra.mul!(y, SDP.full_S, x)
                return y
        end, n)
    eigenvals, _ = symeigs(op, 1; which=:SA, tol=1e-6, maxiter=1000000)
    err4 = max(zero(Tv), -real.(eigenvals[1])) / (1.0 + norm(SDP.C, 2))
    err5 = (SDP.scalars.obj - dot(SDP.λ, SDP.b)) / (1.0 + abs(SDP.scalars.obj) + abs(dot(SDP.λ, SDP.b)))
    err6 = dot(SDP.R, SDP.full_S, SDP.R) / (1.0 + abs(SDP.scalars.obj) + abs(dot(SDP.λ, SDP.b)))
    return [err1, err2, err3, err4, err5, err6]
end


"""
Approximate the minimum eigenvalue of a symmetric matrix `A`.

Perform `q` Lanczos iterations with *a random start vector* to approximate 
the minimum eigenvalue of `A`.
"""
function approx_mineigval_lanczos(
    A::AbstractMatrix{Tv},
    q::Ti,
) where {Ti <: Integer, Tv <: AbstractFloat}
    n::Ti = size(A, 1)
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
        mul!(Av, A, v)
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
    min_eigval, _ = symeigs(B, 1; which=:SA, maxiter=10000, tol=1e-6)
    return min_eigval 
end