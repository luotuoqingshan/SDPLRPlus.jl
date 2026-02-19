function primal_vio(C, As, bs, Rt)
    m = length(bs)
    primal_vio = zeros(Float64, m + 1)
    primal_vio[m+1] = tr(Rt * C * Rt') # objective
    for i in 1:m
        primal_vio[i] = tr(Rt * As[i] * Rt') - bs[i]
    end
    return primal_vio
end

function test_gradient_fd!(data, var, aux)
    r, n = size(var.Rt)
    rt_vec = vec(copy(var.Rt))
    ℒ_scalar(x::Vector) = (copyto!(var.Rt, reshape(x, r, n)); f!(data, var, aux))
    grad_num = FiniteDiff.finite_difference_gradient(ℒ_scalar, copy(rt_vec))
    copyto!(var.Rt, reshape(rt_vec, r, n))
    f!(data, var, aux)
    g!(var, aux)
    grad_ana = vec(var.Gt)
    rel_err = norm(grad_num - grad_ana, Inf) / (1 + norm(grad_ana, Inf))
    @test rel_err < 1e-8
end


@testset "f!, g! and linesearch!" begin
    for (label, prob_fn) in [
        ("MaxCut",         maxcut),
        ("Lovász Theta",   lovasz_theta),
        ("Min. Bisection", minimum_bisection),
        ("Cut Norm",       cutnorm),
    ]
        @testset "$label" begin
            n = 10
            A = make_random_graph(n, 0.6)
            C, As, bs = prob_fn(A)
            r = 3

            data = SDPData(C, As, bs)
            config = BurerMonteiroConfig(σ_0=2.0)
            var = SolverVars(data, r, config)
            aux = SolverAuxiliary(data)

            ℒ_val = f!(data, var, aux)
            @test norm(var.primal_vio - primal_vio(C, As, bs, var.Rt), Inf) < 1e-10

            test_gradient_fd!(data, var, aux)

            g!(var, aux)
            dirt = -1.0 * copy(var.Gt)
            α, 𝓛_val = linesearch!(var, aux, dirt, α_max=1.0)
            axpy!(α, dirt, var.Rt)

            @test norm(var.primal_vio - primal_vio(C, As, bs, var.Rt), Inf) < 1e-10
        end
    end
end


@testset "𝒜t! operator" begin
    # Dense reference: S = Σᵢ var.y[i] * Aᵢ + var.y[m+1] * C
    function At_reference(C, As, var)
        m = length(As)
        S = sum(var.y[i] .* Matrix(As[i]) for i in 1:m)
        S .+= var.y[m+1] .* Matrix(C)
        return S
    end

    for (label, prob_fn) in [
        ("MaxCut (sparse only)",         maxcut),
        ("Lovász Theta (low-rank C)",    lovasz_theta),
        ("Min. Bisection (low-rank As)", minimum_bisection),
    ]
        @testset "$label" begin
            n = 8
            A = make_random_graph(n, 0.5)
            C, As, bs = prob_fn(A)
            r = 2

            data = SDPData(C, As, bs)
            config = BurerMonteiroConfig(σ_0=2.0)
            var = SolverVars(data, r, config)
            aux = SolverAuxiliary(data)

            f!(data, var, aux)

            # Set var.y to known random values, then sync aux.sparse_S
            Random.seed!(42)
            var.y .= randn(length(var.y))
            𝒜t_preprocess!(var, aux)

            S_ref = At_reference(C, As, var)

            # Test left-multiply: 𝒜t!(y, x, aux, var) → y = x * S  (r×n = r×n * n×n)
            Rt = copy(var.Rt)  # r×n
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