# Tests for complex Hermitian SDP support via real embedding (src/complex_embed.jl).
#
# Organisation:
#   1. embed_matrix unit tests (one @testset per supported matrix type):
#      - block structure is correct for each type
#      - output is symmetric
#   2. Inner product preservation: tr(embed(A) * phi(X)) = Re(tr(A * X))
#      for SparseMatrixCSC, Diagonal, SymLowRankMatrix
#   3. embed_hermitian_sdp: b unchanged, matrix types match expectations
#   4. End-to-end: phase retrieval SDP solved to feasibility
#   5. Result format: Rt is complex, has correct size n (not 2n)

# -----------------------------------------------------------------------
# embed_matrix — SparseMatrixCSC
# -----------------------------------------------------------------------
@testset "embed_matrix SparseMatrixCSC: 2n×2n block structure" begin
    for n in [2, 4, 6]
        Random.seed!(n)
        A_dense = randn(ComplexF64, n, n)
        A = (A_dense + A_dense') / 2       # complex Hermitian
        A_sp = sparse(A)
        E = embed_matrix(A_sp)

        @test size(E) == (2n, 2n)
        @test E isa SparseMatrixCSC

        Ed = Matrix(E)
        @test Ed[1:n, 1:n] ≈ real(A) / 2 atol = 1e-12
        @test Ed[1:n, n+1:2n] ≈ -imag(A) / 2 atol = 1e-12
        @test Ed[n+1:2n, 1:n] ≈ imag(A) / 2 atol = 1e-12
        @test Ed[n+1:2n, n+1:2n] ≈ real(A) / 2 atol = 1e-12
        @test issymmetric(Ed)
    end
end

# -----------------------------------------------------------------------
# embed_matrix — Diagonal
# -----------------------------------------------------------------------
@testset "embed_matrix Diagonal: repeated real diagonal" begin
    for n in [2, 4, 6]
        Random.seed!(n + 100)
        d_real = randn(n)
        # A Hermitian diagonal must have purely real entries
        A = Diagonal(complex.(d_real))
        E = embed_matrix(A)

        @test E isa Diagonal
        @test length(E.diag) == 2n
        @test E.diag[1:n] ≈ d_real / 2 atol = 1e-12
        @test E.diag[n+1:2n] ≈ d_real / 2 atol = 1e-12
    end
end

# -----------------------------------------------------------------------
# embed_matrix — SymLowRankMatrix
# -----------------------------------------------------------------------
@testset "embed_matrix SymLowRankMatrix: phi_B D_tilde phi_B^T block structure" begin
    for (n, k) in [(3, 1), (4, 2), (6, 3)]
        Random.seed!(n * 10 + k)
        B = randn(ComplexF64, n, k)
        d = abs.(randn(k)) .+ 0.1    # positive entries
        # SymLowRankMatrix requires D and B to share element type
        A_lr = SymLowRankMatrix(Diagonal(complex.(d)), B)
        E = embed_matrix(A_lr)

        @test E isa SymLowRankMatrix
        @test size(E) == (2n, 2n)

        # Reference: phi(B*D*B^H)/2 in block form
        # (Julia's B' on complex B is the conjugate transpose B^H)
        A_full = B * Diagonal(d) * B'
        expected = [
            real(A_full)/2 -imag(A_full)/2
            imag(A_full)/2 real(A_full)/2
        ]
        @test Matrix(E) ≈ expected atol = 1e-10
    end
end

# -----------------------------------------------------------------------
# Inner product preservation: tr(embed(A) * phi(X)) = Re(tr(A * X))
# -----------------------------------------------------------------------
@testset "Inner product preservation" begin
    for n in [3, 5, 7]
        Random.seed!(n + 200)
        A_dense = randn(ComplexF64, n, n)
        A = (A_dense + A_dense') / 2          # Hermitian cost

        Z = randn(ComplexF64, n, n)
        X = Z * Z' / n                        # PSD primal variable

        phi_X = [
            real(X) -imag(X)
            imag(X) real(X)
        ]

        inner_ref = real(tr(A * X))

        # SparseMatrixCSC
        @test tr(Matrix(embed_matrix(sparse(A))) * phi_X) ≈ inner_ref atol =
            1e-10

        # Diagonal (A's imaginary diagonal entries must be zero for Hermitian)
        d_vals = abs.(randn(n))
        A_diag = Diagonal(complex.(d_vals))
        @test tr(Matrix(embed_matrix(A_diag)) * phi_X) ≈
            real(tr(Matrix(A_diag) * X)) atol = 1e-10

        # SymLowRankMatrix
        B = randn(ComplexF64, n, 2)
        d_lr = abs.(randn(2)) .+ 0.1
        A_lr = SymLowRankMatrix(Diagonal(complex.(d_lr)), B)
        A_lr_dense = B * Diagonal(d_lr) * B'
        @test tr(Matrix(embed_matrix(A_lr)) * phi_X) ≈ real(tr(A_lr_dense * X)) atol =
            1e-10
    end
end

# -----------------------------------------------------------------------
# embed_hermitian_sdp: b is unchanged, C and As are embedded correctly
# -----------------------------------------------------------------------
@testset "embed_hermitian_sdp: b unchanged and matrix dimensions correct" begin
    n = 4
    m = 6
    Random.seed!(42)

    A_dense = randn(ComplexF64, n, n)
    C = sparse((A_dense + A_dense') / 2)
    As = [sparse((M -> (M + M') / 2)(randn(ComplexF64, n, n))) for _ in 1:m]
    b = randn(m)

    C_real, As_real, b_real = embed_hermitian_sdp(C, As, b)

    @test b_real ≈ b
    @test eltype(b_real) == Float64
    @test size(C_real) == (2n, 2n)
    @test all(size(A) == (2n, 2n) for A in As_real)
    @test eltype(C_real) <: Real
    @test all(eltype(A) <: Real for A in As_real)
end

# -----------------------------------------------------------------------
# End-to-end: abstract phase retrieval SDP
#
# Given measurement vectors a_i in C^n and scalar measurements
# b_i = |a_i^H chi|^2, recover chi via:
#
#   minimize   0
#   subject to Re(tr(a_i a_i^H X)) = b_i,  i = 1..m
#              X Hermitian PSD n×n
#
# At optimum X* = chi * chi^H (rank-1, Hermitian PSD). We check:
#   - primal feasibility is achieved (primal_vio <= ptol)
#   - Rt is complex-valued with n columns (not 2n)
#   - X = Rt' * Rt is approximately Hermitian
#   - X is approximately rank-1 (top eigenvalue  >>  second eigenvalue)
# -----------------------------------------------------------------------
@testset "Complex SDP: phase retrieval feasibility" begin
    n = 6
    m = 4 * n
    ptol = 1e-2

    Random.seed!(7)
    chi = randn(ComplexF64, n)
    chi ./= norm(chi)

    # Build rank-1 constraint matrices A_i = a_i * a_i^H as SymLowRankMatrix
    meas_vecs = [randn(ComplexF64, n) for _ in 1:m]
    As = [
        SymLowRankMatrix(Diagonal(ones(ComplexF64, 1)), reshape(a, n, 1)) for
        a in meas_vecs
    ]
    b = [abs2(dot(a, chi)) for a in meas_vecs]

    C = sparse(Diagonal(ones(ComplexF64, n)))   # identity cost

    result = sdplr(
        C, As, b, 1; ptol=ptol, objtol=Inf, printlevel=0, maxtime=60.0
    )

    Rt = result["Rt"]

    # primal feasibility
    @test result["primal_vio"] <= ptol

    # result factor is complex with the original (not doubled) dimension
    @test eltype(Rt) <: Complex
    @test size(Rt, 2) == n

    # X = Rt' * Rt is Hermitian
    X = Rt' * Rt
    @test X ≈ X' atol = 1e-8

    # solution is approximately rank-1
    ev = sort(abs.(eigvals(Hermitian((X + X') / 2))); rev=true)
    @test ev[1] / ev[2] > 10
end

# -----------------------------------------------------------------------
# Result format: other fields remain real scalars
# -----------------------------------------------------------------------
@testset "Complex SDP: result dict types" begin
    n = 4
    m = 2 * n
    Random.seed!(99)

    meas_vecs = [randn(ComplexF64, n) for _ in 1:m]
    chi = randn(ComplexF64, n)
    chi ./= norm(chi)
    As = [
        SymLowRankMatrix(Diagonal(ones(ComplexF64, 1)), reshape(a, n, 1)) for
        a in meas_vecs
    ]
    b = [abs2(dot(a, chi)) for a in meas_vecs]

    C = sparse(Diagonal(ones(ComplexF64, n)))   # identity cost
    result = sdplr(
        C, As, b, 1; ptol=1e-2, objtol=Inf, printlevel=0, maxtime=30.0
    )

    @test eltype(result["Rt"]) <: Complex
    @test eltype(result["Rt0"]) <: Complex
    @test result["obj"] isa Real
    @test result["primal_vio"] isa Real
    @test result["totaltime"] isa Real
    @test result["iter"] isa Integer
    @test size(result["Rt"], 2) == n
    @test size(result["Rt0"], 2) == n
end

# -----------------------------------------------------------------------
# Backward compatibility: real inputs still produce real Rt
# -----------------------------------------------------------------------
@testset "Backward compat: real input gives real Rt" begin
    A = sparse([0.0 1.0; 1.0 0.0])
    C, As, bs = maxcut(A)
    result = sdplr(C, As, bs, 1; ptol=1e-4, objtol=Inf, printlevel=0)

    @test eltype(result["Rt"]) == Float64
    @test result["obj"] ≈ -1.0 atol = 0.05
end
