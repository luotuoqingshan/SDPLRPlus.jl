using LuxurySparse, SparseArrays, LinearAlgebra

super_sparse(I, J, V, n, m) = SparseMatrixCOO(I, J, V, n, m)

"""
Maxcut SDP:              

minimize  - 1/4 ⟨L, X⟩   

s.t.      Diag(X) = 1    

          X ≽ 0    
"""
function maxcut(A::SparseMatrixCSC; Tv=Float64, Ti=Int64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A; dims=2)[:, 1]
    L = sparse(Diagonal(d) - A)
    L .*= Tv(-0.25)
    As = []
    bs = Tv[]
    for i in eachindex(d)
        push!(As, super_sparse(Ti[i], Ti[i], [one(Tv)], n, n))
        push!(bs, one(Tv))
    end
    @info "Max Cut SDP is formed."
    return L, As, bs
end

"""
Lovasz theta SDP:

minimize -⟨1ᵀ1, X⟩

s.t.     Tr(X) = 1

         X_{ij} = 0 for all (i, j) ∈ E

         X ≽ 0
"""
function lovasz_theta(A::SparseMatrixCSC; Tv=Float64, Ti=Int64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    C = SymLowRankMatrix(Diagonal(-ones(Tv, 1)), ones(Tv, (n, 1)))

    As = []
    bs = Tv[]
    for (i, j, _) in zip(findnz(A)...)
        if i < j
            push!(
                As, super_sparse(Ti[i, j], Ti[j, i], [one(Tv), one(Tv)], n, n)
            )
            push!(bs, zero(Tv))
        elseif i == j
            push!(As, super_sparse(Ti[i], Ti[i], [one(Tv)], n, n))
            push!(bs, zero(Tv))
        end
    end
    push!(As, sparse(1.0I, n, n))
    push!(bs, Tv(1))
    @info "Lovasz Theta SDP is formed."
    return C, As, bs
end

"""
Minimum Bisection

minimize 1/4 ⟨L, X⟩

s.t.     Diag(X) = 1

         1ᵀX1 = 0

         X ≽ 0
"""
function minimum_bisection(A::SparseMatrixCSC; Tv=Float64, Ti=Int64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A; dims=2)[:, 1]
    L = sparse(Diagonal(d) - A)
    L ./= 4
    As = []
    bs = Tv[]
    for i in eachindex(d)
        push!(As, super_sparse(Ti[i], Ti[i], [one(Tv)], n, n))
        push!(bs, one(Tv))
    end
    push!(As, SymLowRankMatrix(Diagonal(ones(Tv, 1)), ones(Tv, n, 1)))
    push!(bs, zero(Tv))
    @info "Minimum Bisection SDP is formed."
    return Tv.(L), As, bs
end

function bipartite_matrix(A::SparseMatrixCSC)
    m, n = size(A)
    B = [spzeros(m, m) A; A' spzeros(n, n)]
    return B
end

function cutnorm(A::SparseMatrixCSC; Tv=Float64, Ti=Int64)
    C = bipartite_matrix(A) ./ 2
    As = []
    bs = Tv[]
    for i in 1:size(C, 1)
        push!(As, super_sparse(Ti[i], Ti[i], [one(Tv)], size(C)...))
        push!(bs, one(Tv))
    end
    @info "Cut Norm SDP is formed."
    return -Tv.(C), As, bs
end

function extend_sparse(A, new_n, new_m)
    I, J, V = findnz(A)
    return sparse(I, J, V, new_n, new_m)
end

mu_conductance_ub(volG, mu) = (1 - mu) / (mu * volG)
mu_conductance_lb(volG, mu) = mu / ((1 - mu) * volG)

"""
μ-Conductance SDP:              

minimize  ⟨L, X⟩   

s.t.      ⟨D, X⟩ = 1

          ⟨ddᵀ, X⟩ = 0

          Diag(X) ≤ (1 - μ)/(μ Vol(G)) 

          Diag(X) ≥ μ/((1 - μ) Vol(G)) 

          X ≽ 0
"""
function mu_conductance(A::SparseMatrixCSC, mu; Tv=Float64, Ti=Int64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A; dims=2)[:, 1]
    D = sparse(Diagonal(d))
    volG = sum(d)
    L = sparse(Diagonal(d) - A)
    As = []
    bs = Tv[]

    padded_d = [d; zeros(Tv, 2 * n)]
    padded_D = extend_sparse(D, 3 * n, 3 * n)
    padded_L = extend_sparse(L, 3 * n, 3 * n)

    push!(As, padded_D)
    push!(bs, 1)

    push!(As, SymLowRankMatrix(Diagonal(ones(Tv, 1)), reshape(padded_d, :, 1)))
    push!(bs, zero(Tv))

    ub = mu_conductance_ub(volG, mu)
    lb = mu_conductance_lb(volG, mu)

    for i in eachindex(d)
        push!(
            As,
            super_sparse(
                Ti[i, i+n], Ti[i, i+n], [one(Tv), one(Tv)], 3 * n, 3 * n
            ),
        )
        push!(bs, Tv(ub))
    end
    for i in eachindex(d)
        push!(
            As,
            super_sparse(
                Ti[i, i+2*n], Ti[i, i+2*n], [one(Tv), -one(Tv)], 3 * n, 3 * n
            ),
        )
        push!(bs, Tv(lb))
    end
    return Tv.(padded_L), As, bs
end

"""
Maxcut SDP:              

minimize  - 1/4 ⟨L, X⟩   

s.t.      0.99 ≤ Diag(X) ≤ 1    

          X ≽ 0    
"""
function relaxed_maxcut(A::SparseMatrixCSC; Tv=Float64, Ti=Int64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A; dims=2)[:, 1]
    L = sparse(Diagonal(d) - A)
    L .*= Tv(-0.25)
    padded_L = extend_sparse(L, 3 * n, 3 * n)
    As = []
    bs = Tv[]
    for i in eachindex(d)
        push!(
            As,
            super_sparse(
                Ti[i, i+n], Ti[i, i+n], [one(Tv), one(Tv)], 3 * n, 3 * n
            ),
        )
        push!(bs, one(Tv))
    end
    for i in eachindex(d)
        push!(
            As,
            super_sparse(
                Ti[i, i+2*n], Ti[i, i+2*n], [one(Tv), -one(Tv)], 3 * n, 3 * n
            ),
        )
        push!(bs, Tv(0.99))
    end
    return Tv.(padded_L), As, bs
end

"""
μ-Conductance SDP:              

minimize  ⟨L, X⟩   

s.t.      ⟨D, X⟩ = 1

          ⟨ddᵀ, X⟩ = 0

          Diag(X) ≤ (1 - μ)/(μ Vol(G)) 

          Diag(X) ≥ μ/((1 - μ) Vol(G)) 

          X ≽ 0
"""
function mu_conductance_reformulated(
    A::SparseMatrixCSC, mu; Tv=Float64, Ti=Int64
)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A; dims=2)[:, 1]
    D = sparse(Diagonal(d))
    volG = sum(d)
    L = sparse(Diagonal(d) - A)

    As = []
    bs = Tv[]

    ddt_norm = norm(d, 2)^2
    D_norm = norm(D, 2)
    padded_d = [d; zeros(Tv, 2 * n)]
    padded_D = extend_sparse(D, 3 * n, 3 * n)
    padded_L = extend_sparse(L, 3 * n, 3 * n)

    push!(As, padded_D)
    push!(bs, one(Tv))

    push!(As, SymLowRankMatrix(Diagonal([one(Tv)]), reshape(padded_d, :, 1)))
    push!(bs, zero(Tv))

    ub = mu_conductance_ub(volG, mu)
    lb = mu_conductance_lb(volG, mu)

    for i in eachindex(d)
        push!(
            As,
            super_sparse(
                Ti[i, i+n], Ti[i, i+n], [one(Tv), one(Tv)], 3 * n, 3 * n
            ),
        )
        push!(bs, Tv(ub))
    end
    for i in eachindex(d)
        push!(
            As,
            super_sparse(
                Ti[i+n, i+2*n], Ti[i+n, i+2*n], [one(Tv), one(Tv)], 3 * n, 3 * n
            ),
        )
        push!(bs, Tv(ub - lb))
    end
    return Tv.(padded_L), As, bs
end

"""
μ-Conductance SDP (native inequality formulation, n×n):

minimize  ⟨L, X⟩

s.t.      ⟨D, X⟩ = 1
          ⟨ddᵀ, X⟩ = 0
          X[i,i] ≤ ub    (represented as ⟨eᵢeᵢᵀ, X⟩ ≤ ub,   constraint_type = true)
          X[i,i] ≥ lb    (flipped to    ⟨-eᵢeᵢᵀ, X⟩ ≤ -lb,  constraint_type = true)
          X ≽ 0

Returns (C, As, bs, constraint_types) for use with sdplr(...; constraint_types=ct).
"""
function mu_conductance_native(A::SparseMatrixCSC, mu; Tv=Float64, Ti=Int64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A; dims=2)[:, 1]
    D = sparse(Diagonal(d))
    volG = sum(d)
    L = sparse(Diagonal(d) - A)

    As = []
    bs = Tv[]
    ct = Bool[]

    D_norm = norm(D, 2)
    d_max = maximum(d)
    dd_norm = norm(d, 2)^2

    # Equality 1: ⟨D, X⟩ = 1
    push!(As, D)
    push!(bs, one(Tv))
    push!(ct, false)

    # Equality 2: ⟨ddᵀ, X⟩ = 0  (rank-1 low-rank matrix)
    push!(
        As, SymLowRankMatrix(Diagonal([Tv(D_norm / dd_norm)]), reshape(d, :, 1))
    )
    push!(bs, zero(Tv))
    push!(ct, false)

    ub = mu_conductance_ub(volG, mu)
    lb = mu_conductance_lb(volG, mu)

    # Inequality ≤ ub: X[i,i] ≤ ub  →  ⟨eᵢeᵢᵀ, X⟩ ≤ ub
    for i in 1:n
        push!(As, super_sparse(Ti[i], Ti[i], [Tv(D_norm)], n, n))
        push!(bs, Tv(ub * D_norm))
        push!(ct, true)
    end

    # Inequality ≥ lb: X[i,i] ≥ lb  →  flip: ⟨-eᵢeᵢᵀ, X⟩ ≤ -lb
    for i in 1:n
        push!(As, super_sparse(Ti[i], Ti[i], [-Tv(D_norm)], n, n))
        push!(bs, Tv(-lb * D_norm))
        push!(ct, true)
    end

    return Tv.(L), As, bs, ct
end