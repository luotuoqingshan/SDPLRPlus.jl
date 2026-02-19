@testset "Cut Norm" begin
  @testset "primal violation evaluation f!" begin
    n = 10
    A = make_random_graph(n, 0.6)

    C, As, bs = cutnorm(A)
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
