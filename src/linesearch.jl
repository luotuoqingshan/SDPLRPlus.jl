"""
Exact line search for minimizing the augmented Lagrangian
"""
function linesearch!(
    var::SolverVars{Ti,Tv}, aux, Dt::AbstractArray{Tv}; α_max=one(Tv)
) where {Ti<:Integer,Tv}
    m = length(var.primal_vio)-1
    # evaluate 𝓐(RDᵀ + DRᵀ)
    RD_dt = @elapsed begin
        𝒜!(var.A_RD, aux, var.Rt, Dt)
    end
    # remember we divide it by 2 in Aoper, now scale back
    var.A_RD .*= 2.0
    # evaluate 𝓐(DDᵀ)
    DD_dt = @elapsed begin
        𝒜!(var.A_DD, aux, Dt, Dt)
    end
    @debug "RD_dt, DD_dt" RD_dt, DD_dt

    biquadratic = zeros(5)
    cubic = zeros(4)

    # p0 = ⟨ C, RRᵀ⟩           = var.obj[] 
    # p1 = ⟨ C, (RDᵀ + DRᵀ)⟩   = C_RD
    # p2 = ⟨ C, DDᵀ⟩           = C_DD
    # (-q0) = 𝓐(RRᵀ) - b      = var.primal_vio
    # q1 = 𝓐(RDᵀ + DRᵀ)       = 𝓐_RD
    # q2 = 𝓐(DDᵀ)             = 𝓐_DD

    # f(x) = a x⁴ + b x³ + c x² + d x + e 
    # a = σ / 2 * ||q2||²
    # b = σ * q1ᵀ * q2
    # c = p2 - λᵀ * q2 + σ (-q0)ᵀ * q2 + σ /2 ||q1||²
    # d = p1 - λᵀ * q1 + σ (-q0)ᵀ * q1
    # e = p0 - λᵀ * (-q0) + σ / 2 * ||-q0||²
    p0 = var.obj[]
    p1 = var.A_RD[m+1]
    p2 = var.A_DD[m+1]
    neg_q0 = @view var.primal_vio[1:m]
    q1 = @view var.A_RD[1:m]
    q2 = @view var.A_DD[1:m]
    σ = var.σ[]

    biquadratic[1] = (p0 - dot(var.λ, neg_q0) + σ * dot(neg_q0, neg_q0) / 2)

    # in principle biquadratic[2] should equal to 
    # the inner product between direction and gradient
    # thus it should be negative

    biquadratic[2] = (p1 - dot(var.λ, q1) + σ * dot(neg_q0, q1))

    biquadratic[3] = (p2 - dot(var.λ - σ * neg_q0, q2) + σ * dot(q1, q1) / 2)

    biquadratic[4] = σ * dot(q1, q2)

    biquadratic[5] = σ * dot(q2, q2) / 2

    cubic[1] = 1.0 * biquadratic[2]

    if cubic[1] > eps()
        error("Error: cubic[1] = $(cubic[1]) should be less than 0.")
    end

    cubic[2] = 2.0 * biquadratic[3]

    cubic[3] = 3.0 * biquadratic[4]

    cubic[4] = 4.0 * biquadratic[5]

    if abs(cubic[4]) < eps()
        # got a quadractic function
        # error("Error: cubic[4] is zero, got a quadratic function")
        quadratic = cubic[1:3]
        quadratic ./= quadratic[3]
        f = Polynomial(biquadratic)
        df = Polynomial(quadratic)

        f0 = biquadratic[1] # f(α=0) 
        α_star = 0.0 # optimal α
        f_star = f0 # optimal f(α)

        Roots = PolynomialRoots.roots(quadratic)
        push!(Roots, α_max)

    else
        cubic ./= cubic[4]
        f = Polynomial(biquadratic)
        df = Polynomial(cubic)

        f0 = biquadratic[1] # f(α=0) 
        α_star = 0.0 # optimal α
        f_star = f0 # optimal f(α)

        Roots = PolynomialRoots.roots(cubic)
        push!(Roots, α_max)
    end

    for i in eachindex(Roots)
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

    # update the primal violation and function value
    # notice that 
    # 𝓐((R + αD)(R + αD)ᵀ) =   
    # 𝓐(RRᵀ) + α 𝓐(RDᵀ + DRᵀ) + α² 𝓐(DDᵀ)
    @. var.primal_vio += α_star * (α_star * var.A_DD + var.A_RD)
    var.obj[] = var.primal_vio[m+1]

    return α_star, f_star
end
