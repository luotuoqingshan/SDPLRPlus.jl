@testset "Max Cut" begin
    @testset "A toy graph with 2 vertices" begin
        A = sparse([0.0 1.0;
            1.0 0.0])
        C, As, bs = maxcut(A)
        n = size(A, 1)
        r = 1
        res = sdplr(C, As, bs, r; 
            fprec=0.0, objtol=1e-8, ptol=1e-8, prior_trace_bound=Float64(n))
        @test res["obj"] ≈ -1 
    end
    @testset "a different σ_0" begin
        A = sparse([0.0 1.0;
            1.0 0.0])
        C, As, bs = maxcut(A)
        n = size(A, 1)
        r = 1
        res = sdplr(C, As, bs, r; σ_0=10.0,
            fprec=0.0, objtol=1e-8, ptol=1e-8, prior_trace_bound=Float64(n))
        @test res["obj"] ≈ -1 
    end
    @testset "using customized init function" begin
        A = sparse([0.0 1.0;
            1.0 0.0])
        C, As, bs = maxcut(A)
        function init_func(data, r, sigma)
            return randn(r, size(data.C, 1)) * sqrt(sigma), zeros(size(data.b, 1))
        end
        n = size(A, 1)
        r = 1
        res = sdplr(C, As, bs, r; init_func=init_func, init_args=(10.0,),
            fprec=0.0, objtol=1e-8, ptol=1e-8, prior_trace_bound=Float64(n))
        @test res["obj"] ≈ -1 
    end
    @testset "test f! g! and linesearch!" begin
        n = 10
        A = make_random_graph(n, 0.6)

        C, As, bs = maxcut(A)
        r = 3

        # Set up the problem data structures
        data = SDPData(C, As, bs)
        config = BurerMonteiroConfig(σ_0=2.0)
        var = SolverVars(data, r, config)
        aux = SolverAuxiliary(data)

        # Call f! to evaluate primal violation
        ℒ_val = f!(data, var, aux)
        @test norm(var.primal_vio - primal_vio(C, As, bs, var.Rt), Inf) < 1e-10

        # test gradient g! vs FiniteDiff
        test_gradient_fd!(data, var, aux)

        # simulate linesearch step and check primal violation
        g!(var, aux)
        dirt = -1.0 * copy(var.Gt)
        α, 𝓛_val = linesearch!(var, aux, dirt, α_max=1.0)
        axpy!(α, dirt, var.Rt)

        @test norm(var.primal_vio - primal_vio(C, As, bs, var.Rt), Inf) < 1e-10
    end
end
