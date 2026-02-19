# End-to-end solver test for the Minimum Bisection SDP.
# K₂ has one cut edge, so ¼⟨L,X⟩ at optimum equals 1.
@testset "Minimum Bisection" begin
    @testset "A toy graph with 2 vertices" begin
        A = sparse([
            0.0 1.0;
            1.0 0.0
        ])
        C, As, bs = minimum_bisection(A)
        n = size(A, 1)
        r = 1
        res = sdplr(
            C,
            As,
            bs,
            r;
            fprec=0.0,
            objtol=1e-4,
            ptol=1e-4,
            prior_trace_bound=Float64(n),
        )
        @test (res["obj"]-1) / (1 + abs(res["obj"])) < 1e-4
    end
end
