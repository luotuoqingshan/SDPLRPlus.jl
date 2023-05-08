using Polynomials
using PolynomialRoots
include("dataoper.jl")
include("structs.jl")

"""
Exact line search for minimizing the augmented Lagrangian
"""
function linesearch!(
    BM::BurerMonteiro{Tv},
    SDP::SDPProblem{Ti, Tv, TC, TCons},
    D::Matrix{Tv};
    α_max = one(Tv),
    update = true,
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons}
    # evaluate 𝓐(RDᵀ + DRᵀ)

    C_RD, 𝓐_RD = Aoper(SDP, BM.R, D, same=false, calcobj=true)
    # remember we divide it by 2 in Aoper, now scale back
    𝓐_RD .*= 2.0
    C_RD *= 2.0
    # evaluate 𝓐(DDᵀ)
    C_DD, 𝓐_DD = Aoper(SDP, D, D, same=true, calcobj=true)

    biquadratic = zeros(5)
    cubic = zeros(4)

    # p0 = ⟨ C, RRᵀ⟩           = BM.obj 
    # p1 = ⟨ C, (RDᵀ + DRᵀ)⟩   = C_RD
    # p2 = ⟨ C, DDᵀ⟩           = C_DD
    # (-q0) = 𝓐(RRᵀ) - b      = BM.primal_vio
    # q1 = 𝓐(RDᵀ + DRᵀ)       = 𝓐_RD
    # q2 = 𝓐(DDᵀ)             = 𝓐_DD
     
    # f(x) = a x⁴ + b x³ + c x² + d x + e 
    # a = σ / 2 * ||q2||²
    # b = σ * q1ᵀ * q2
    # c = p2 - λᵀ * q2 + σ (-q0)ᵀ * q2 + σ /2 ||q1||²
    # d = p1 - λᵀ * q1 + σ (-q0)ᵀ * q1
    # e = p0 - λᵀ * (-q0) + σ / 2 * ||-q0||²

    m = SDP.m
    biquadratic[1] = (BM.obj - dot(BM.λ, BM.primal_vio) + 
        0.5 * BM.σ * dot(BM.primal_vio, BM.primal_vio))
    
    # in principle biquadratic[2] should equal to 
    # the inner product between direction and gradient
    # thus it should be negative
    biquadratic[2] = (C_RD - dot(BM.λ, 𝓐_RD) + 
        BM.σ * dot(BM.primal_vio, 𝓐_RD))  
    

    biquadratic[3] = (C_DD - dot(BM.λ, 𝓐_DD) + 
        BM.σ * dot(BM.primal_vio, 𝓐_DD) + 
        0.5 * BM.σ * dot(𝓐_RD, 𝓐_RD))

    biquadratic[4] = BM.σ * dot(𝓐_DD, 𝓐_RD)

    biquadratic[5] = 0.5 * BM.σ * dot(𝓐_DD, 𝓐_DD)

    cubic[1] = 1.0 * biquadratic[2]

    if cubic[1] > eps()
        println("Warning: cubic[1] = $(cubic[1]) should be less than 0.")
        return 0
    end

    cubic[2] = 2.0 * biquadratic[3]

    cubic[3] = 3.0 * biquadratic[4]

    cubic[4] = 4.0 * biquadratic[5]

    if abs(cubic[4]) < eps()
        println("Warning: cubic[4] is zero, got a quadratic function")
        return 0
    end

    cubic ./= cubic[4]
    f = Polynomial(biquadratic)
    df = Polynomial(cubic)

    f0 = biquadratic[1] # f(α=0) 
    α_star = 0.0 # optimal α
    f_star = f0 # optimal f(α)

    Roots = PolynomialRoots.roots(cubic)
    push!(Roots, α_max)

    for i = eachindex(Roots)
        # only examine real roots in [0, α_max]
        @show real(Roots[i])
        if (abs(imag(Roots[i])) >= eps())    
            continue
        end
        root = real(Roots[i])
        if (root < 0) || (root > α_max)
            continue
        end
        f_α = f(root)
        if f_α < f_star
            f_star = f_α
            α_star = root 
        end
    end

    @show α_star, f_star

    if update == true 
        # notice that 
        # 𝓐((R + αD)(R + αD)ᵀ) =   
        # 𝓐(RRᵀ) + α 𝓐(RDᵀ + DRᵀ) + α² 𝓐(DDᵀ)
        @. BM.primal_vio += α_star * (α_star * 𝓐_DD + 𝓐_RD)
        BM.obj += α_star * (α_star * C_DD + C_RD)
    end

    return α_star, f_star 
end