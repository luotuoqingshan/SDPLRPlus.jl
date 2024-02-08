"""
Maxcut SDP:              

minimize  - 1/4 ⟨L, X⟩   

s.t.      Diag(X) = 1    

          X ≽ 0    
"""
function maxcut(A::SparseMatrixCSC; Tv=Float64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A, dims=2)[:, 1]
    L = sparse(Diagonal(d) - A)
    L .*= Tv(-0.25)
    As = []
    bs = [] 
    for i in eachindex(d)
        push!(As, sparse([i], [i], [one(Tv)], n, n))
        push!(bs, one(Tv))
    end
    return L, As, bs
end


"""
Lovasz theta SDP:

minimize -⟨1ᵀ1, X⟩

s.t.     Tr(X) = 1

         X_{ij} = 0 for all (i, j) ∈ E

         X ≽ 0
"""
function lovasz_theta_SDP(A::SparseMatrixCSC; Tv=Float64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    C = LowRankMatrix(Diagonal(-ones(Tv, 1)), ones(Tv, (n, 1)))

    As = []
    bs = []
    for (i, j, _) in zip(findnz(A)...)
        if i < j
            push!(As, sparse([i, j], [j, i], [one(Tv), one(Tv)], n, n))
            push!(bs, zero(Tv))
        elseif i == j
            push!(As, sparse([i], [i], [one(Tv)], n, n))
            push!(bs, zero(Tv))
        end
    end
    push!(As, sparse(Matrix{Tv}(I, n, n)))
    push!(bs, one(Tv))
    return C, As, bs
end


"""
Minimum Bisection

minimize 1/4 ⟨L, X⟩

s.t.     Diag(X) = 1

         1ᵀX1 = 0

         X ≽ 0
"""
function minimum_bisection(A::SparseMatrixCSC; Tv=Float64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A, dims=2)[:, 1]
    L = sparse(Diagonal(d) - A)
    L ./= 4
    As = []
    bs = Tv[]
    for i in eachindex(d)
        push!(As, sparse([i], [i], [one(Tv)], n, n))
        push!(bs, one(Tv))
    end
    push!(As, LowRankMatrix(Diagonal(ones(Tv, 1)), ones(Tv, n, 1)))
    push!(bs, zero(Tv))
    return Tv.(L) , As, bs
end


