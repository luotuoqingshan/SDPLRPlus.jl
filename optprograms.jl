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


function maxcut_write_sdplr(A::AbstractMatrix, filepath::String; Tv=Float64)
    n = size(A, 1) 
    d = sum(A, dims=2)[:, 1]
    L = sparse(Diagonal(d) - A)
    C = -Tv.(L)
    open(filepath, "w") do f
        write(f, "$n\n") # number of constraint matrices
        write(f, "1\n")  # number of blocks in the SDP
        write(f, "$n\n") # sizes of the blocks
        write(f, ("1.0 "^n)*"\n") # b
        write(f, "1.0\n")
        triuC = triu(C)
        write(f, "0 1 s $(nnz(triuC))\n")
        for (i, j, v) in zip(findnz(triuC)...)
            write(f, "$i $j $v\n")
        end
        for i = 1:n
            write(f, "$i 1 s 1\n")
            write(f, "$i $i 1.0\n")
        end
    end
end


function maxcut_write_sdpa(A::AbstractMatrix, filepath::String; Tv=Float64)
    n = size(A, 1) 
    d = sum(A, dims=2)[:, 1]
    L = sparse(Diagonal(d) - A)
    C = -Tv.(L)
    open(filepath, "w") do f
        write(f, "$n\n") # number of constraint matrices
        write(f, "1\n")  # number of blocks in the SDP
        write(f, "$n\n") # sizes of the blocks
        write(f, ("1.0 "^n)*"\n") # b
        triuC = triu(C)
        for (i, j, v) in zip(findnz(triuC)...)
            write(f, "0 1 $i $j $v\n")
        end
        for i = 1:n
            write(f, "$i 1 $i $i 1.0\n")
        end
    end
end

function lovasz_theta_SDP(A::AbstractMatrix; Tv=Float64)
end

A = load_gset(pwd()*"/SDPLR-jl/data/Gset/G12") 
C, As, bs = maxcut(A)
res = sdplr(C, As, bs, 10)

filepath = pwd()*"/data/G12.sdpa"
maxcut_write_sdpa(A, filepath)