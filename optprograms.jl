include("structs.jl")
include("sdplr.jl")
include("readdata.jl")
using LinearAlgebra, SparseArrays

function maxcut(A::AbstractMatrix; Tv=Float64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A, dims=2)[:, 1]
    L = sparse(Diagonal(d) - A)
    As = []
    bs = zeros(Tv, n) 
    for i in eachindex(d)
        push!(As, sparse([i], [i], [one(Tv)], n, n))
        bs[i] = one(Tv) 
    end
    return -Tv.(L), As, bs
end

A = load_gset(pwd()*"SDPLR-jl/data/Gset/G1") 
C, As, bs = maxcut(A)
res = sdplr(C, As, bs, 10)

function lovasz_theta_SDP()
end