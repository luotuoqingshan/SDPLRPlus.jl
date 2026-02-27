# Unit tests for the three core operations used inside the solver loop:
#   f!          — evaluates the augmented Lagrangian and primal violations
#   g!          — computes the gradient of the augmented Lagrangian w.r.t. R
#   linesearch! — exact quartic line search along a descent direction
#   𝒜t!         — adjoint operator S = Σᵢ yᵢ Aᵢ + y_{m+1} C, applied to R

# Dense reference for primal violations: [⟨Aᵢ, RRᵀ⟩ - bᵢ; ⟨C, RRᵀ⟩]
function primal_vio(C, As, bs, Rt)
    m = length(bs)
    primal_vio = zeros(Float64, m + 1)
    primal_vio[m+1] = tr(Rt * C * Rt') # objective
    for i in 1:m
        primal_vio[i] = tr(Rt * As[i] * Rt') - bs[i]
    end
    return primal_vio
end

# Checks analytic gradient from g! against finite differences; passes if rel-err < 1e-8.
function test_gradient_fd!(data, var, aux)
    r, n = size(var.Rt)
    rt_vec = vec(copy(var.Rt))
    function ℒ_scalar(x::Vector)
        return (copyto!(var.Rt, reshape(x, r, n)); f!(data, var, aux))
    end
    grad_num = FiniteDiff.finite_difference_gradient(ℒ_scalar, copy(rt_vec))
    copyto!(var.Rt, reshape(rt_vec, r, n))
    f!(data, var, aux)
    g!(var, aux)
    grad_ana = vec(var.Gt)
    rel_err = norm(grad_num - grad_ana, Inf) / (1 + norm(grad_ana, Inf))
    @test rel_err < 1e-8
end

# 7 problem types × 12 (n,p,r) combos × 3 @test calls ≈ 252 tests
@testset "f!, g! and linesearch!" begin
    for (label, prob_fn) in [
        ("MaxCut", maxcut),
        ("Lovász Theta", lovasz_theta),
        ("Min. Bisection", minimum_bisection),
        ("Cut Norm", cutnorm),
        ("μ-Conductance μ=0.01", A -> mu_conductance(A, 0.01)),
        ("μ-Conductance μ=0.05", A -> mu_conductance(A, 0.05)),
        ("μ-Conductance μ=0.1", A -> mu_conductance(A, 0.1)),
    ]
        @testset "$label" begin
            for (seed, (n, p, r)) in
                enumerate(Iterators.product([5, 8, 12], [0.4, 0.7], [2, 3]))
                @testset "n=$n p=$p r=$r" begin
                    Random.seed!(seed)
                    A = make_random_graph(n, p)
                    C, As, bs = prob_fn(A)

                    data = SDPData(C, As, bs)
                    config = BurerMonteiroConfig(; σ_0=2.0)
                    var = SolverVars(data, r, config)
                    aux = SolverAuxiliary(data)

                    f!(data, var, aux)
                    @test norm(
                        var.primal_vio_raw - primal_vio(C, As, bs, var.Rt), Inf
                    ) < 1e-10

                    test_gradient_fd!(data, var, aux)

                    g!(var, aux)
                    dirt = -1.0 * copy(var.Gt)
                    α, 𝓛_val = linesearch!(var, aux, dirt; α_max=1.0)
                    axpy!(α, dirt, var.Rt)

                    @test norm(
                        var.primal_vio_raw - primal_vio(C, As, bs, var.Rt), Inf
                    ) < 1e-10
                end
            end
        end
    end
end

# Dense reference for primal_vio: equality violations are free, inequality violations are capped at 0.
function primal_vio_reference(pv, constraint_types)
    cap = copy(pv[1:length(constraint_types)])
    for i in eachindex(constraint_types)
        if constraint_types[i]  # inequality ≤
            cap[i] = max(cap[i], 0.0)
        end
    end
    return cap
end

# Tests for inequality constraints using mu_conductance_ineq (n×n, no slack-variable lift).
# Checks: primal_vio_raw, primal_vio, gradient (vs finite differences).
@testset "f!, g! with inequality constraints" begin
    for mu in [0.01, 0.05, 0.1]
        @testset "μ-Conductance-ineq μ=$mu" begin
            for (seed, (n, p, r)) in
                enumerate(Iterators.product([5, 8, 12], [0.4, 0.7], [2, 3]))
                @testset "n=$n p=$p r=$r" begin
                    Random.seed!(seed)
                    A = make_random_graph(n, p)
                    C, As, bs, constraint_types = mu_conductance_ineq(A, mu)

                    data = SDPData(C, As, bs, constraint_types)
                    config = BurerMonteiroConfig(; σ_0=2.0)
                    var = SolverVars(data, r, config)
                    aux = SolverAuxiliary(data)

                    f!(data, var, aux)
                    pv = primal_vio(C, As, bs, var.Rt)
                    @test norm(var.primal_vio_raw - pv, Inf) < 1e-10

                    pvc_ref = primal_vio_reference(pv, constraint_types)
                    @test norm(var.primal_vio - pvc_ref, Inf) < 1e-10

                    test_gradient_fd!(data, var, aux)
                end
            end
        end
    end
end

# Dense reference: S = Σᵢ var.y[i] * Aᵢ + var.y[m+1] * C
function At_reference(C, As, var)
    m = length(As)
    S = sum(var.y[i] .* Matrix(As[i]) for i in 1:m)
    S .+= var.y[m+1] .* Matrix(C)
    return S
end

# 6 problem types × 12 (n,p,r) combos × 2 @test calls = 144 tests
@testset "𝒜t! operator" begin
    for (label, prob_fn) in [
        ("MaxCut (sparse only)", maxcut),
        ("Lovász Theta (low-rank C)", lovasz_theta),
        ("Min. Bisection (low-rank As)", minimum_bisection),
        ("μ-Conductance μ=0.01 (mixed formats)", A -> mu_conductance(A, 0.01)),
        ("μ-Conductance μ=0.05 (mixed formats)", A -> mu_conductance(A, 0.05)),
        ("μ-Conductance μ=0.1 (mixed formats)", A -> mu_conductance(A, 0.1)),
    ]
        @testset "$label" begin
            for (seed, (n, p, r)) in
                enumerate(Iterators.product([5, 8, 12], [0.4, 0.7], [2, 3]))
                @testset "n=$n p=$p r=$r" begin
                    Random.seed!(seed)
                    A = make_random_graph(n, p)
                    C, As, bs = prob_fn(A)
                    # Rebind n to the SDP dimension (mu_conductance pads to 3n).
                    n = size(C, 1)

                    data = SDPData(C, As, bs)
                    config = BurerMonteiroConfig(; σ_0=2.0)
                    var = SolverVars(data, r, config)
                    aux = SolverAuxiliary(data)

                    f!(data, var, aux)

                    Random.seed!(seed + 100)
                    var.y .= randn(length(var.y))
                    𝒜t_preprocess!(var, aux)

                    S_ref = At_reference(C, As, var)

                    # Test left-multiply: 𝒜t!(y, x, aux, var) → y = x * S  (r×n = r×n * n×n)
                    Rt = copy(var.Rt)
                    y_left = zeros(eltype(Rt), r, n)
                    𝒜t!(y_left, Rt, aux, var)
                    @test norm(y_left - Rt * S_ref, Inf) < 1e-10

                    # Test right-multiply: 𝒜t!(y, aux, x, var) → y = S * x  (n×r = n×n * n×r)
                    x_right = randn(n, r)
                    y_right = zeros(n, r)
                    𝒜t!(y_right, aux, x_right, var)
                    @test norm(y_right - S_ref * x_right, Inf) < 1e-10
                end
            end
        end
    end

    for mu in [0.01, 0.05, 0.1]
        @testset "μ-Conductance-ineq μ=$mu (n×n, inequality)" begin
            for (seed, (n, p, r)) in
                enumerate(Iterators.product([5, 8, 12], [0.4, 0.7], [2, 3]))
                @testset "n=$n p=$p r=$r" begin
                    Random.seed!(seed)
                    A = make_random_graph(n, p)
                    C, As, bs, constraint_types = mu_conductance_ineq(A, mu)
                    n = size(C, 1)

                    data = SDPData(C, As, bs, constraint_types)
                    config = BurerMonteiroConfig(; σ_0=2.0)
                    var = SolverVars(data, r, config)
                    aux = SolverAuxiliary(data)

                    f!(data, var, aux)

                    Random.seed!(seed + 100)
                    var.y .= randn(length(var.y))
                    𝒜t_preprocess!(var, aux)

                    S_ref = At_reference(C, As, var)

                    Rt = copy(var.Rt)
                    y_left = zeros(eltype(Rt), r, n)
                    𝒜t!(y_left, Rt, aux, var)
                    @test norm(y_left - Rt * S_ref, Inf) < 1e-10

                    x_right = randn(n, r)
                    y_right = zeros(n, r)
                    𝒜t!(y_right, aux, x_right, var)
                    @test norm(y_right - S_ref * x_right, Inf) < 1e-10
                end
            end
        end
    end
end
