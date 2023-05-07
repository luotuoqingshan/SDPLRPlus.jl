using Test
@testset "structs ops' tests" begin
    include("structs.jl")
    @testset "LowRankMatrix tests" begin
        ntests = 100
        for i = 1:ntests
            n = rand(50:100) 
            s = rand(1:20) 
            r = rand(1:20) 
            A = LowRankMatrix(Diagonal(randn(s)), randn(n, s), r)
            Adense = A.B * A.D * A.Bt
            @test norm(A, 2) ≈ norm(Adense, 2)
            @test norm(A, Inf) ≈ norm(Adense, Inf)

            X = randn(n, r)
            @test norm(A * X - Adense * X) < 1e-10 
            Y = randn(n, r)
            @test abs(dot_xTAx(A, X) - dot(X, Adense, X)) < 1e-10
            @test abs(dot(X, A, Y) - dot(X, Adense, Y)) < 1e-10
        end
    end

    @testset "UnitLowRankMatrix tests" begin
        ntests = 100
        for i = 1:ntests
            n = rand(50:100)
            s = rand(1:20)
            r = rand(1:20)
            A = UnitLowRankMatrix(randn(n, s), r)
            Adense = A.B * A.Bt
            @test norm(A, 2) ≈ norm(Adense, 2)
            @test norm(A, Inf) ≈ norm(Adense, Inf)

            X = randn(n, r)
            @test norm(A * X - Adense * X) < 1e-10 
            Y = randn(n, r)
            @test abs(dot_xTAx(A, X) - dot(X, Adense, X)) < 1e-10
            @test abs(dot(X, A, Y) - dot(X, Adense, Y)) < 1e-10
        end
    end

    include("sdplr.jl")
    @testset "MaxCut test" begin
        A = [0 1;
            1 0]
        n = size(A, 1)
        d = sum(A, dims=2)[:, 1]
        L = sparse(Diagonal(d) - A)
        As = []
        bs = Float64[]
        for i in eachindex(d)
            ei = zeros(n, 1)
            ei[i, 1] = 1
            push!(As, LowRankMatrix(Diagonal([1.0]), ei))
            push!(bs, 1.0)
        end
        r = 1
        res = sdplr(-Float64.(L), As, bs, r)
        @test abs(res["obj"] - (-4)) < 1e-4
    end
end

