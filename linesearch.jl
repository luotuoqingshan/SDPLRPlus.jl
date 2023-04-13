using Polynomials
using PolynomialRoots
include("dataoper.jl")
include("structures.jl")

"""
Exact line search for minimizing the augmented Lagrangian
"""
function linesearch!(
    pdata::ProblemData,
    algdata::AlgorithmData,
    R::Matrix{Float64},
    D::Matrix{Float64};
    alpha_max = 1.0,
    update = 1,
)
    # evaluate \\cal{A}(RD^T + DR^T)
    calA_RD = Aoper(pdata, R, D, same=0, obj=1, large=large)
    # remember we divide it by 2 in Aoper, now scale back
    calA_RD .*= 2.0
    # evaluate \\cal{A}(DD^T)
    calA_DD = Aoper(pdata, D, D, same=1, obj=1, large=large)

    biquadratic = zeros(5)
    cubic = zeros(4)

    # p0 = C \dot (RR^T)           = algdata.vio[end]
    # p1 = C \dot (RD^T + DR^T)    = calA_RD[end]
    # p2 = C \dot (DD^T)           = calA_DD[end]
    # (-q0) = \calA(RR^T) - b      = algdata.vio[1:m]
    # q1 = \calA(RD^T + DR^T)      = calA_RD[1:m]
    # q2 = \calA(DD^T)             = calA_DD[1:m]

    # f(x) = a x^4 + b x^3 + c x^2 + d x + e 
    # a = sigma / 2 * ||q2||^2
    # b = sigma * q1' * q2
    # c = p2 - lambda' * q2 + sigma (-q0)' * q2 + sigma/2 ||q1||^2
    # d = p1 - lambda' * q1 + sigma (-q0)' * q1
    # e = p0 - lambda * (-q0) + sigma / 2 * ||-q0||^2

    biquadratic[1] = algdata.vio[end] - algdata.lambda' * algdata.vio[1:pdata.m] + 
        0.5 * algdata.sigma * algdata.vio[1:pdata.m]' * vio_RR[1:pdata.m]
    
    biquadratic[2] = calA_RD[end] - algdata.lambda' * calA_RD[1:pdata.m] + 
        algdata.sigma * algdata.vio[1:pdata.m]' * calA_RD[1:pdata.m]  

    biquadratic[3] = calA_DD[end] - algdata.lambda' * calA_DD[1:pdata.m] + 
        algdata.sigma * algdata.vio[1:pdata.m]' * calA_DD[1:pdata.m] + 
        0.5 * algdata.sigma * calA_RD[1:pdata.m]' * calA_RD[1:pdata.m]

    biquadratic[4] = algdata.sigma * calA_DD[1:pdata.m]' * calA_RD[1:pdata.m]

    biquadratic[5] = 0.5 * algdata.sigma * calA_DD[1:pdata.m]' * calA_DD[1:pdata.m]

    cubic[1] = 1.0 * biquadratic[2]

    if cubic[1] > eps()
        println("Warning: cubic[1] = $(cubic[1]) should be less than 0.")
        exit(0)
    end

    cubic[2] = 2.0 * biquadratic[3]

    cubic[3] = 3.0 * biquadratic[4]

    cubic[4] = 4.0 * biquadratic[5]

    if abs(cubic[4]) < eps()
        println("Warning: cubic[4] is zero, got a quadratic function")
        exit(0)
    end

    cubic ./= cubic[4]
    f = Polynomial(biquadratic)
    df = Polynomial(cubic)

    f0 = biquadratic[0] # f(alpha=0) 
    alpha_star = 0 # optimal alpha
    f_star = f0 # optimal f(alpha)

    Roots = roots(df)
    push!(Roots, alpha_max)

    for i = eachindex(Roots)
        # only examine real roots in [0, alpha_max]
        if (Roots[i] < 0) || (Roots[i] > alpha_max) || (abs(imag(Roots[i])) >= eps()) 
            continue
        end
        f_alpha = f(Roots[i])
        if f_alpha < f_star
            f_star = f_alpha
            alpha_star = Roots[i]
        end
    end

    if update
        # notice that 
        # \calA((R + alpha D)(R + alpha D)^T) = 
        # \calA(RR^T) + alpha \calA(RD^T + DR^T) + alpha^2 \calA(DD^T)
        algdata.vio += alpha_star * (alpha_star * calA_DD + calA_RD)
    end

    return alpha_star, f_star 
end