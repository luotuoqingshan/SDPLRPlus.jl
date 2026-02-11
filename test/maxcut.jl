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
end
