include("structs.jl")


"""
This function computes the augmented Lagrangian value, 
    𝓛(R, λ, σ) = Tr(C RRᵀ) - λᵀ(𝓐(RRᵀ) - b) + σ/2 ||𝓐(RRᵀ) - b||^2
"""
function lagrangval!(
    BM::BurerMonteiro{Ti, Tv}, 
    SDP::SDPProblem{Ti, Tv, TC}, 
    ) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    # apply the operator 𝓐 to RRᵀ and 
    # potentially compute the objective function value
    BM.scalars.obj = Aoper!(BM.primal_vio, SDP.global_UVt, SDP, BM.R, BM.R; same=true)
    BM.primal_vio .-= SDP.b 
    return (BM.scalars.obj - dot(BM.λ, BM.primal_vio)
           + BM.scalars.σ * dot(BM.primal_vio, BM.primal_vio) / 2) 
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


"""
This function computes the gradient of the augmented Lagrangian
"""
function gradient!(
    BM::BurerMonteiro{Ti, Tv},
    SDP::SDPProblem{Ti, Tv, TC},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    m = SDP.m
    @. BM.y = -(BM.λ - BM.scalars.σ * BM.primal_vio)
    fill!(BM.G, zero(Tv))
    n, r = size(BM.R)


    fill!(SDP.S_nzval, zero(Tv))
    for i = 1:SDP.n_spase_matrices
        ind = SDP.sparse_As_global_inds[i]
        coeff = ind == 0 ? one(Tv) : BM.y[ind]
        for j = SDP.agg_A_ptr[i]:(SDP.agg_A_ptr[i + 1] - 1)
            SDP.S_nzval[SDP.agg_A_nzind[j]] += SDP.agg_A_nzval_one[j] * coeff
        end
    end


    #constraint_grad!(BM.G, S, SDP.C, SDP.indC, BM.R, one(Tv))
    #for (i, A) in enumerate(SDP)
    #    constraint_grad!(BM.G, S, A, SDP.indAs[i], BM.R, y[i])
    #end
    @inbounds for col = 1:SDP.n 
        for nzi = SDP.XS_colptr[col]:(SDP.XS_colptr[col + 1] - 1)
            row = SDP.XS_rowval[nzi]
            @simd for k = axes(BM.G, 2) 
                BM.G[row, k] += SDP.S_nzval[nzi] * BM.R[col, k]
            end
            if row != col
                @simd for k = axes(BM.G, 2) 
                    BM.G[col, k] += SDP.S_nzval[nzi] * BM.R[row, k]
                end
            end
        end
    end
    #mul!(BM.G, S, BM.R, one(Tv), one(Tv))
    lmul!(Tv(2), BM.G)
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
    BM::BurerMonteiro{Ti, Tv},
    SDP::SDPProblem{Ti, Tv, TC},
    normC::Tv,
    normb::Tv,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    #L_val_dt = @elapsed begin
    𝓛_val = lagrangval!(BM, SDP)
    #end
    #grad_dt = @elapsed begin
    gradient!(BM, SDP)
    #end
    stationarity = norm(BM.G, 2) / (1.0 + normC)
    primal_vio = norm(BM.primal_vio, 2) / (1.0 + normb)
    #@show L_val_dt, grad_dt
    return (𝓛_val, stationarity, primal_vio)
end
