using LinearAlgebra, SparseArrays


"""
Maxcut SDP: 
minimize  - 1/4 ⟨L, X⟩
s.t.      Diag(X) = 1
          X ≽ 0
"""
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


#"""
#Generate the SDPLR format for the maxcut SDP.
#"""
#function maxcut_write_sdplr(A::AbstractMatrix, filepath::String; Tv=Float64)
#    n = size(A, 1) 
#    d = sum(A, dims=2)[:, 1]
#    L = sparse(Diagonal(d) - A)
#    C = -Tv.(L)
#    open(filepath, "w") do f
#        write(f, "$n\n") # number of constraint matrices
#        write(f, "1\n")  # number of blocks in the SDP
#        write(f, "$n\n") # sizes of the blocks
#        write(f, ("1.0 "^n)*"\n") # b
#        write(f, "1.0\n")
#        triuC = triu(C)
#        write(f, "0 1 s $(nnz(triuC))\n")
#        for (i, j, v) in zip(findnz(triuC)...)
#            write(f, "$i $j $v\n")
#        end
#        for i = 1:n
#            write(f, "$i 1 s 1\n")
#            write(f, "$i $i 1.0\n")
#        end
#    end
#end
#
#
#"""
#Generate the SDPA format for the maxcut SDP.
#"""
#function maxcut_write_sdpa(A::AbstractMatrix, filepath::String; Tv=Float64)
#    n = size(A, 1) 
#    d = sum(A, dims=2)[:, 1]
#    L = sparse(Diagonal(d) - A)
#    C = Tv.(L)
#    open(filepath, "w") do f
#        write(f, "$n\n") # number of constraint matrices
#        write(f, "1\n")  # number of blocks in the SDP
#        write(f, "$n\n") # sizes of the blocks
#        write(f, ("1.0 "^n)*"\n") # b
#        triuC = triu(C)
#        for (i, j, v) in zip(findnz(triuC)...)
#            write(f, "0 1 $i $j $v\n")
#        end
#        for i = 1:n
#            write(f, "$i 1 $i $i 1.0\n")
#        end
#    end
#end
#
#
#function maxcut_write_sol_in(R, λ, filepath::String)
#    n, r = size(R)
#    open(filepath, "w") do f
#        write(f, "dual variable $n\n")
#        for i = 1:n
#            write(f, "$(λ[i])\n")
#        end
#        write(f, "primal variable 1 s $n $r $r\n")
#        for j = axes(R, 2) 
#            for i = axes(R, 1)
#                write(f, "$(R[i, j])\n")
#            end
#        end
#        write(f, "special majiter 0\n")
#        write(f, "special iter 0\n")
#        write(f, "special lambdaupdate 0")
#        write(f, "special CG 0\n")
#        write(f, "special curr_CG 0\n")
#        write(f, "special totaltime 0\n")
#        write(f, "special sigma $(1.0/n)\n") 
#        write(f, "special scale 1.0\n")
#    end
#end


"""
Lovasz theta SDP:
maximize ⟨1ᵀ1, X⟩
s.t.     Tr(X) = 1
         X_{ij} = 0 for all (i, j) ∈ E
         X ≽ 0
"""
function lovasz_theta_SDP(A::AbstractMatrix; Tv=Float64)

end


"""
Minimum Bisection
minimize 1/4 ⟨L, X⟩
s.t.     Diag(X) = 1
         1ᵀX1 = 0
         X ≽ 0
"""
function minimum_bisection(A::AbstractMatrix; Tv=Float64)
    @assert A == A' "Only undirected graphs supported now."
    n = size(A, 1)
    d = sum(A, dims=2)[:, 1]
    L = sparse(Diagonal(d) - A)
    L ./= 4
    As = []
    bs = zeros(Tv, n + 1) 
    for i in eachindex(d)
        push!(As, sparse([i], [i], [one(Tv)], n, n))
        bs[i] = one(Tv) 
    end
    push!(As, UnitLowRankMatrix(ones(Tv, n, 1)))
    bs[n + 1] = zero(Tv)
    return Tv.(L) , As, bs
end


#"""
#Write initial solution for SDPLR-1.03-beta 
#"""
#function write_initial_solution(
#    R::Matrix{Tv}, 
#    λ::Vector{Tv}, 
#    filepath::String,
#) where {Tv <: AbstractFloat}
#    n, r = size(R)
#    m = length(λ)
#    open(filepath, "w") do f
#        write(f, "dual variable $m\n")
#        for i = 1:m
#            write(f, "$(λ[i])\n")
#        end
#        write(f, "primal variable 1 s $n $r $r\n")
#        for j = axes(R, 2) 
#            for i = axes(R, 1)
#                write(f, "$(R[i, j])\n")
#            end
#        end
#        write(f, "special majiter 0\n")
#        write(f, "special iter 0\n")
#        write(f, "special lambdaupdate 0")
#        write(f, "special CG 0\n")
#        write(f, "special curr_CG 0\n")
#        write(f, "special totaltime 0\n")
#        write(f, "special sigma $(1.0/n)\n") 
#        write(f, "special scale 1.0\n")
#    end
#end


function write_problem_sdpa(
    C::AbstractMatrix{Tv},
    As::TCons,
    bs::Vector{Tv},
    filepath::String,
) where {Tv <: AbstractFloat, TCons}
    n = size(C, 1) 
    m = length(As)
    open(filepath, "w") do f
        write(f, "$m\n") # number of constraint matrices
        write(f, "1\n")  # number of blocks in the SDP
        write(f, "$n\n") # sizes of the blocks
        for i = 1:m
            write(f, "$(bs[i])")
        end
        write(f, "\n")
        triuC = triu(C)
        for (i, j, v) in zip(findnz(triuC)...)
            write(f, "0 1 $i $j $v\n")
        end
        for i = 1:m
            triuAi = triu(As[i])
            for (r, c, v) in zip(findnz(triuAi)...)
                write(f, "$i 1 $r $c $v\n")
            end
        end
    end
end