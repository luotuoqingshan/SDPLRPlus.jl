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
    C = Tv.(L)
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


function maxcut_write_sol_in(R, λ, filepath::String)
    n, r = size(R)
    open(filepath, "w") do f
        write(f, "dual variable $n\n")
        for i = 1:n
            write(f, "$(λ[i])\n")
        end
        write(f, "primal variable 1 s $n $r $r\n")
        for j = axes(R, 2) 
            for i = axes(R, 1)
                write(f, "$(R[i, j])\n")
            end
        end
        write(f, "special majiter 0\n")
        write(f, "special iter 0\n")
        write(f, "special lambdaupdate 0")
        write(f, "special CG 0\n")
        write(f, "special curr_CG 0\n")
        write(f, "special totaltime 0\n")
        write(f, "special sigma $(1.0/n)\n") 
        write(f, "special scale 1.0\n")
    end
end

function lovasz_theta_SDP(A::AbstractMatrix; Tv=Float64)
end
