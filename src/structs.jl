using Random
using Test
using SparseArrays
using LinearAlgebra
import LinearAlgebra: dot, size, show, norm

"""
Low rank representation of constraint matrices
written as BDBᵀ, since usually B is really thin,
so storing Bᵀ as well doesn't cost too much more storage.
"""
struct LowRankMatrix{T} <: AbstractMatrix{T}
    D::Diagonal{T}
    B::Matrix{T}
    Bt::Matrix{T}
    extra::Matrix{T}
    function LowRankMatrix(
        D::Diagonal{T},
        B::Matrix{T},
        r::Ti=0,
    )where {T, Ti <: Integer}
        n, s = size(B) 
        return new{T}(D, B, Matrix(B'), zeros(s,r))
    end
end


struct UnitLowRankMatrix{T} <: AbstractMatrix{T}
    B::Matrix{T}
    Bt::Matrix{T}
    extra::Matrix{T}
    function UnitLowRankMatrix(
        B::Matrix{T},
        r::Ti=0,
    )where {T, Ti <: Integer}
        n, s = size(B) 
        return new{T}(B, Matrix(B'), zeros(s,r))
    end
end


#LowRankMatrix(D::Diagonal{T}, B::Matrix{T}) where T = LowRankMatrix(D, B, Matrix(B'), zeros(0,0))
#function LowRankMatrix(
#    D::Diagonal{Tv}, 
#    B::Matrix{Tv}, 
#    r::Ti,
#) where {Ti <: Integer, Tv <: AbstractFloat} 
#    n, s = size(B)
#    return LowRankMatrix(D, B, Matrix(B'), zeros(s,r)) 
#end
#
#UnitLowRankMatrix(B::Matrix{T}) where T = UnitLowRankMatrix(B, Matrix(B'), zeros(0,0))
#function UnitLowRankMatrix(
#    B::Matrix{Tv},
#    r::Ti,
#) where {Ti <: Integer, Tv <: AbstractFloat}
#    n, s = size(B)
#    return UnitLowRankMatrix(B, Matrix(B'), zeros(s,r)) 
#end

size(A::LowRankMatrix) = (n = size(A.B, 1); (n, n))
size(A::UnitLowRankMatrix) = (n = size(A.B, 1); (n, n))
Base.getindex(A::LowRankMatrix, i::Int, j::Int) = @view(A.Bt[:, i])' * A.D * @view(A.Bt[:, j])
Base.getindex(A::UnitLowRankMatrix, i::Int, j::Int) = @view(A.Bt[:, i])' * @view(A.Bt[:, j])


function show(io::IO, mime::MIME{Symbol("text/plain")}, A::LowRankMatrix)
    summary(io, A) 
    println(io)
    println(io, "LowRankMatrix of form BDBᵀ.")
    println(io, "B factor:")
    show(io, mime, A.B)
    println(io, "\nD factor:")
    show(io, mime, A.D)
end


function show(io::IO, mime::MIME{Symbol("text/plain")}, A::UnitLowRankMatrix)
    summary(io, A) 
    println(io)
    println(io, "UnitLowRankMatrix of form BBᵀ.")
    println(io, "B factor:")
    show(io, mime, A.B)
end


function norm(
    A::LowRankMatrix{Tv},
    p::Real,
) where {Tv <: AbstractFloat}
    # norm is usually not performance critical
    # so we don't do too much preallocation
    n = size(A.B, 1)
    tmpv = zeros(n)
    if p in [2, Inf]
        # BDBᵀ 
        U = A.B * A.D
        res = zero(Tv) 
        @inbounds for i in axes(A.Bt, 2)
            mul!(tmpv, U, @view(A.Bt[:, i]))
            if p == Inf 
                res = max(res, norm(tmpv, p))
            else
                res += norm(tmpv, p)^2
            end
        end
        return p == Inf ? res : sqrt(res)
    else
        error("undefined norm for Constraint")
    end
end


function LinearAlgebra.mul!(
    Y::AbstractMatrix{Tv},
    A::LowRankMatrix{Tv},
    X::AbstractMatrix{Tv},
    ) where {Tv <: AbstractFloat}
    n, _ = size(A.B)
    if size(Y, 1) != n || size(X, 1) != n || size(Y, 2) != size(X, 2) 
        throw(DimensionMismatch("dimension mismatch"))
    end
    # A.Bt: s x n , X: n x r, A.extra: s x r
    mul!(A.extra, A.Bt, X)
    lmul!(A.D, A.extra)
    mul!(Y, A.B, A.extra)
end


#using LinearAlgebra, SparseArrays, BenchmarkTools
#function mymul!(
#    Y::AbstractMatrix{Tv},
#    A::SparseMatrixCSC{Tv},
#    X::AbstractMatrix{Tv},
#    α::Tv,
#    β::Tv,
#    ) where {Tv <: AbstractFloat}
#    lmul!(β, Y)
#    Yt = Y'
#    Xt = X'
#    if (size(Y, 1) != size(A, 1) || size(X, 1) != size(A, 2) || size(Y, 2) != size(X, 2))
#        throw(DimensionMismatch("dimension mismatch"))
#    end
#    for (x, y, v) in zip(findnz(A)...)
#        @view(Yt[:, x]) .+= α * v * @view(Xt[:, y]) 
#    end
#end


function dot_xTAx(
    A::LowRankMatrix{Tv},
    X::AbstractMatrix{Tv},
    ) where {Tv <: AbstractFloat}
    mul!(A.extra, A.Bt, X)
    return dot(A.extra, A.D, A.extra)
end


function dot(
    X::AbstractMatrix{Tv},
    A::LowRankMatrix{Tv},
    Y::AbstractMatrix{Tv},
    ) where {Tv <: AbstractFloat}
    return dot(X, A * Y)
end


function norm(A::UnitLowRankMatrix{Tv}, p::Real) where {Tv <: AbstractFloat}
    n, _ = size(A.B)
    tmpv = zeros(n)
    if p in [2, Inf]
        # BBᵀ 
        res = zero(Tv) 
        @inbounds for i in axes(A.Bt, 2)
            mul!(tmpv, A.B, @view(A.Bt[:, i]))
            if p == Inf 
                res = max(res, norm(tmpv, p))
            else
                res += norm(tmpv, p)^2
            end
        end
        return p == Inf ? res : sqrt(res)
    else
        error("undefined norm for Constraint")
    end
end


function LinearAlgebra.mul!(
    Y::AbstractMatrix{Tv}, 
    A::UnitLowRankMatrix{Tv}, 
    X::AbstractMatrix{Tv},
    ) where {Tv <: AbstractFloat}
    n = size(A.B, 1)
    if size(Y, 1) != n || size(X, 1) != n || size(Y, 2) != size(X, 2) 
        throw(DimensionMismatch("dimension mismatch"))
    end
    mul!(A.extra, A.Bt, X)
    mul!(Y, A.B, A.extra)
end


function dot_xTAx(
    A::UnitLowRankMatrix{Tv},
    X::AbstractMatrix{Tv}
    ) where {Tv <: AbstractFloat}
    mul!(A.extra, A.Bt, X)
    return dot(A.extra, A.extra)
end


function dot(
    X::AbstractMatrix{Tv},
    A::UnitLowRankMatrix{Tv},
    Y::AbstractMatrix{Tv}
    ) where {Tv <: AbstractFloat}
    return dot(X, A * Y)
end


# fall back function for other matrices
dot_xTAx(A::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T} = dot(X, A, X)


constraint_eval_UTAU(
    A::AbstractMatrix{T}, 
    X::AbstractMatrix{T}, 
    Xt::Adjoint{T}
) where {T} = dot_xTAx(A, X)

constraint_eval_UTAV(
    A::AbstractMatrix{T}, 
    U::AbstractMatrix{T},
    Ut::Adjoint{T}, 
    V::AbstractMatrix{T},
    Vt::Adjoint{T},
) where {T} = (dot(U, A, V) + dot(V, A, U)) / 2


function constraint_eval_UTAU(
    A::SparseMatrixCSC{T}, 
    X::AbstractMatrix{T}, 
    Xt::Adjoint{T}
) where{T}
    res = zero(T)
    @inbounds for(x, y, v) in zip(findnz(A)...)
        res += v * dot(@view(Xt[:, x]), @view(Xt[:, y]))
    end
    return res
end


function constraint_eval_UTAV(
    A::SparseMatrixCSC{T}, 
    U::AbstractMatrix{T}, 
    Ut::Adjoint{T}, 
    V::AbstractMatrix{T}, 
    Vt::Adjoint{T}) where{T}
    res = zero(T)
    @inbounds for(x, y, v) in zip(findnz(A)...)
        res += (v * (dot(@view(Ut[:, x]), @view(Vt[:, y])) 
                + dot(@view(Vt[:, x]), @view(Ut[:, y]))))
    end
    return res / 2
end


function constraint_grad!(
    G::AbstractMatrix{T},
    S::SparseMatrixCSC{T},
    #Gt::AbstractMatrix{T},
    A::AbstractMatrix{T},
    ind::Vector{Ti},
    R::AbstractMatrix{T},
    #Rt::Adjoint{T},
    α::T,
    ) where {T, Ti}
    mul!(G, A, R, α, one(T))
end


function constraint_grad!(
    G::AbstractMatrix{T},
    S::SparseMatrixCSC{T},
    #Gt::AbstractMatrix{T},
    A::SparseMatrixCSC{T},
    ind::Vector{Ti},
    R::AbstractMatrix{T},
    #Rt::Adjoint{T},
    α::T,
    ) where {T, Ti}
    @inbounds @simd for i in axes(ind)
        S.nzval[ind[i]] += α * A.nzval[i]
    end
end


function constraint_grad!(
    G::AbstractMatrix{T},
    S::SparseMatrixCSC{T},
    #Gt::AbstractMatrix{T},
    A::Diagonal{T},
    ind::Vector{Ti},
    R::AbstractMatrix{T},
    #Rt::Adjoint{T},
    α::T,
    ) where {T, Ti}
    @inbounds @simd for i in axes(ind)
        S.nzval[ind[i]] += α * A.diag[i]
    end
end
#
#
#function constraint_grad!(
#    G::AbstractMatrix{T},
#    Gt::AbstractMatrix{T},
#    A::SparseMatrixCSC{T},
#    R::AbstractMatrix{T},
#    Rt::Adjoint{T},
#    α::T,
#    ) where {T}
#    if (size(G, 1) != size(A, 1) || size(G, 2) != size(R, 2) || size(A, 2) != size(R, 1))
#        throw(DimensionMismatch("dimension mismatch"))
#    end
#    @inbounds for (x, y, v) in zip(findnz(A)...)
#        @view(Gt[:, x]) .+= α * v * @view(Rt[:, y])
#    end
#end


#TODO: support block-wise data
struct SDPProblem{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons, Tind} 
    # number of constraints
    m::Ti       
    # list of matrices which are sparse/dense/low-rank/diagonal
    constraints::TCons
    # cost matrix
    C::TC
    # right-hand side b
    b::Vector{Tv}
    # aggregated matrix for sparse constraints
    aggsparse::SparseMatrixCSC{Tv}
    indC::Tind
    indAs::Vector{Tind}
end


function Base.:iterate(SDP::SDPProblem, state=1)
    return state > SDP.m ? nothing : (SDP.constraints[state], state + 1)
end


function Base.:getindex(SDP::SDPProblem, i::Int)
    1 <= i <= SDP.m || throw(BoundsError(SDP, i))
    return SDP.constraints[i]
end

function Base.:length(SDP::SDPProblem)
    return SDP.m
end

mutable struct BurerMonterioScalarVars{Tv<:AbstractFloat}
    # penalty parameter
    σ::Tv
    # objective
    obj::Tv
    # timing
    starttime::Tv
    endtime::Tv
    dual_time::Tv
    primal_time::Tv
end

struct BurerMonteiro{Tv<:AbstractFloat}
    # primal variables X = RR^T
    R::Matrix{Tv}
    # gradient w.r.t. R
    G::Matrix{Tv}
    # dual variables
    λ::Vector{Tv}
    # violation of constraints
    primal_vio::Vector{Tv}
    # penalty parameter
    vars::BurerMonterioScalarVars{Tv}
end

