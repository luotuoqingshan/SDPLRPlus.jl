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
    op = ArpackSimpleFunctionOp(
        (y, x) -> begin
                LinearAlgebra.mul!(y, SDP.full_S, x)
                return y
        end, n)
    eigenvals, eigenvecs = symeigs(op, 1; which=:SA, ncv=min(100, n), maxiter=1000000)
    @show real.(eigenvals[1])
    duality_gap = (SDP.scalars.obj - dot(SDP.λ, SDP.b) + SDP.scalars.σ/2 * dot(SDP.primal_vio, AX + SDP.b)
           - trace_bound * real.(eigenvals[1]))     
    rel_duality_gap = duality_gap / (1 + SDP.scalars.obj)
    return duality_gap, rel_duality_gap 
end
