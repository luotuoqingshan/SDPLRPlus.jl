@testset "Minimum Bisection" begin
    @testset "A toy graph with 2 vertices" begin
        A = sparse([0.0 1.0;
            1.0 0.0])
        C, As, bs = minimum_bisection(A)
        n = size(A, 1)
        r = 1
        res = sdplr(C, As, bs, r; 
            fprec=0.0, objtol=1e-4, ptol=1e-4, prior_trace_bound=Float64(n))
        @test (res["obj"]-1) / (1 + abs(res["obj"])) < 1e-4 
    end
    @testset "primal violation evaluation f!" begin
        # Generate a random symmetric graph A
        n = 10  # Number of vertices
        p = 0.6  # Edge probability
        A_dense = rand(n, n)
        A_dense = (A_dense .+ A_dense') / 2  # Make symmetric
        A_dense = Float64.(A_dense .> p)  # Sparsify
        A_dense[diagind(A_dense)] .= 0.0  # Remove self-loops
        A = sparse(A_dense)
        
        C, As, bs = minimum_bisection(A)
        r = 3  # Random rank
        
        # Set up the problem data structures
        data = SDPData(C, As, bs)
        config = BurerMonteiroConfig(σ_0=2.0)
        var = SolverVars(data, r, config)
        aux = SolverAuxiliary(data)
        
        m = length(var.λ)
        # Call f! to evaluate primal violation
        ℒ_val = f!(data, var, aux)
        my_primal_vio = zeros(Float64, m + 1)
        my_primal_vio[m+1] = tr(var.Rt * C * var.Rt') # objective
    
        for i in 1:m
            my_primal_vio[i] = tr(var.Rt * As[i] * var.Rt') - bs[i]
        end
        @test norm(var.primal_vio - my_primal_vio, Inf) < 1e-10

        # simulate linesearch step and check primal violation
        g!(var, aux)
        dirt = -1.0 * copy(var.Gt)
        α, 𝓛_val = linesearch!(var, aux, dirt, α_max=1.0) 
        axpy!(α, dirt, var.Rt)
        
        my_primal_vio[m+1] = tr(var.Rt * C * var.Rt') # objective

        for i in 1:m
            my_primal_vio[i] = tr(var.Rt * As[i] * var.Rt') - bs[i]
        end
        @test norm(var.primal_vio - my_primal_vio, Inf) < 1e-10
    end
end
