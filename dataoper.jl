include("structs.jl")


"""
This function computes the augmented Lagrangian value, 
    𝓛(R, λ, σ) = Tr(C RRᵀ) - λᵀ(𝓐(RRᵀ) - b) + σ/2 ||𝓐(RRᵀ) - b||^2
"""
function lagrangval!(
    BM::BurerMonteiro{Tv}, 
    SDP::SDPProblem{Ti, Tv, TC, TCons}, 
    ) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons}
    # apply the operator 𝓐 to RRᵀ and 
    # potentially compute the objective function value
    BM.obj, _ = Aoper!(BM.primal_vio, SDP, BM.R, BM.R; same=true, calcobj=true)
    BM.primal_vio .-= SDP.b 
    return (BM.obj - dot(BM.λ, BM.primal_vio)
           + BM.σ * dot(BM.primal_vio, BM.primal_vio) / 2) 
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
    SDP::SDPProblem{Ti, Tv, TC, TCons},
    U::Matrix{Tv},
    V::Matrix{Tv};
    same::Bool=true,
    calcobj::Bool=true,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons}
    fill!(𝓐_UV, zero(eltype(𝓐_UV)))
    obj = zero(Tv) 
    # deal with sparse and diagonal constraints first
    base = 0
    # store results of 𝓐(UVᵀ + VUᵀ)/2
    if same   
        @inbounds for (i, A) in enumerate(SDP) 
            𝓐_UV[i] = constraint_eval_UTAU(A, U)
        end
    else
        @inbounds for (i, A) in enumerate(SDP) 
            𝓐_UV[i] = constraint_eval_UTAV(A, U, V) 
        end
    end

    # if calcobj = true, deal with objective function value
    if calcobj 
        if same
            obj = constraint_eval_UTAU(SDP.C, U) 
        else
            obj = constraint_eval_UTAV(SDP.C, U, V)
        end
    end
    return (obj, 𝓐_UV)
end


function Aoper(
    SDP::SDPProblem{Ti, Tv, TC, TCons},
    U::Matrix{Tv},
    V::Matrix{Tv};
    same::Bool=true,
    calcobj::Bool=true,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons}
    𝓐_UV = zeros(eltype(SDP.C), length(SDP))
    obj, _ = Aoper!(𝓐_UV, SDP, U, V, same=same, calcobj=calcobj)
    return (obj, 𝓐_UV)
end

"""
This function computes the gradient of the augmented Lagrangian
"""
function gradient!(
    BM::BurerMonteiro{Tv},
    SDP::SDPProblem{Ti, Tv, TC, TCons},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons}
    m = SDP.m
    y = similar(BM.λ)
    @. y = -(BM.λ - BM.σ * BM.primal_vio)

    mul!(BM.G, SDP.C, BM.R)
    @inbounds for (i, A) in enumerate(SDP) 
        mul!(BM.G, A, BM.R, y[i], one(eltype(BM.G)))
    end
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
    BM::BurerMonteiro{Tv},
    SDP::SDPProblem{Ti, Tv, TC, TCons},
    normC::Tv,
    normb::Tv,
) where {Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons}
    𝓛_val = lagrangval!(BM, SDP)
    gradient!(BM, SDP)
    stationarity = norm(BM.G, 2) / (1.0 + normC)
    primal_vio = norm(BM.primal_vio, 2) / (1.0 + normb)
    return (𝓛_val, stationarity, primal_vio)
end
