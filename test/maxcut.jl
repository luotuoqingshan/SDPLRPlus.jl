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
end
