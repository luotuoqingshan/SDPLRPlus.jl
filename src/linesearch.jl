"""
Exact line search for minimizing the augmented Lagrangian
"""
function linesearch!(
    SDP::SDPProblem{Ti, Tv, TC},
    D::Matrix{Tv};
    α_max = one(Tv),
    update = true,
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}}
    # evaluate 𝓐(RDᵀ + DRᵀ)
    C_RD = Aoper!(SDP.A_RD, SDP.UVt, SDP, SDP.R, D; same=false)
    # remember we divide it by 2 in Aoper, now scale back
    SDP.A_RD .*= 2.0
    C_RD *= 2.0
    # evaluate 𝓐(DDᵀ)
    C_DD = Aoper!(SDP.A_DD, SDP.UVt, SDP, D, D; same=true)

    biquadratic = zeros(5)
    cubic = zeros(4)

    # p0 = ⟨ C, RRᵀ⟩           = SDP.obj 
    # p1 = ⟨ C, (RDᵀ + DRᵀ)⟩   = C_RD
    # p2 = ⟨ C, DDᵀ⟩           = C_DD
    # (-q0) = 𝓐(RRᵀ) - b      = SDP.primal_vio
    # q1 = 𝓐(RDᵀ + DRᵀ)       = 𝓐_RD
    # q2 = 𝓐(DDᵀ)             = 𝓐_DD
     
    # f(x) = a x⁴ + b x³ + c x² + d x + e 
    # a = σ / 2 * ||q2||²
    # b = σ * q1ᵀ * q2
    # c = p2 - λᵀ * q2 + σ (-q0)ᵀ * q2 + σ /2 ||q1||²
    # d = p1 - λᵀ * q1 + σ (-q0)ᵀ * q1
    # e = p0 - λᵀ * (-q0) + σ / 2 * ||-q0||²

    m = SDP.m
    biquadratic[1] = (SDP.obj - dot(SDP.λ, SDP.primal_vio) + 
        0.5 * SDP.σ * dot(SDP.primal_vio, SDP.primal_vio))
    
    # in principle biquadratic[2] should equal to 
    # the inner product between direction and gradient
    # thus it should be negative
    biquadratic[2] = (C_RD - dot(SDP.λ, SDP.A_RD) + 
        SDP.σ * dot(SDP.primal_vio, SDP.A_RD))  
    

    biquadratic[3] = (C_DD - dot(SDP.λ, SDP.A_DD) + 
        SDP.σ * dot(SDP.primal_vio, SDP.A_DD) + 
        0.5 * SDP.σ * dot(SDP.A_RD, SDP.A_RD))

    biquadratic[4] = SDP.σ * dot(SDP.A_DD, SDP.A_RD)

    biquadratic[5] = 0.5 * SDP.σ * dot(SDP.A_DD, SDP.A_DD)

    cubic[1] = 1.0 * biquadratic[2]

    if cubic[1] > eps()
        error("Error: cubic[1] = $(cubic[1]) should be less than 0.")
    end

    cubic[2] = 2.0 * biquadratic[3]

    cubic[3] = 3.0 * biquadratic[4]

    cubic[4] = 4.0 * biquadratic[5]

    if abs(cubic[4]) < eps()
        error("Error: cubic[4] is zero, got a quadratic function")
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

    if update == true 
        # notice that 
        # 𝓐((R + αD)(R + αD)ᵀ) =   
        # 𝓐(RRᵀ) + α 𝓐(RDᵀ + DRᵀ) + α² 𝓐(DDᵀ)
        @. SDP.primal_vio += α_star * (α_star * SDP.A_DD + SDP.A_RD)
        SDP.obj += α_star * (α_star * C_DD + C_RD)
    end

    return α_star, f_star 
end