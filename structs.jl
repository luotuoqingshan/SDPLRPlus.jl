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
end


struct UnitLowRankMatrix{T} <: AbstractMatrix{T}
    B::Matrix{T}
    Bt::Matrix{T}
    extra::Matrix{T}
end


LowRankMatrix(D::Diagonal{T}, B::Matrix{T}) where T = LowRankMatrix(D, B, Matrix(B'), zeros(0,0))
function LowRankMatrix(
    D::Diagonal{Tv}, 
    B::Matrix{Tv}, 
    r::Ti,
) where {Ti <: Integer, Tv <: AbstractFloat} 
    n, s = size(B)
    return LowRankMatrix(D, B, Matrix(B'), zeros(s,r)) 
end

UnitLowRankMatrix(B::Matrix{T}) where T = UnitLowRankMatrix(B, Matrix(B'), zeros(0,0))
function UnitLowRankMatrix(
    B::Matrix{Tv},
    r::Ti,
) where {Ti <: Integer, Tv <: AbstractFloat}
    n, s = size(B)
    return UnitLowRankMatrix(B, Matrix(B'), zeros(s,r)) 
end

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


constraint_eval_UTAU(A::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T} = dot_xTAx(A, X)
constraint_eval_UTAU(A::LowRankMatrix{T}, X::AbstractMatrix{T}) where {T} = dot_xTAx(A, X) 
constraint_eval_UTAU(A::UnitLowRankMatrix{T}, X::AbstractMatrix{T}) where {T} = dot_xTAx(A, X)

constraint_eval_UTAV(A::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T} = (dot(U, A, V) + dot(V, A, U)) / 2
constraint_eval_UTAV(A::LowRankMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T} = (dot(U, A, V) + dot(V, A, U)) / 2
constraint_eval_UTAV(A::UnitLowRankMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T} = (dot(U, A, V) + dot(V, A, U)) / 2


#TODO: support block-wise data
struct SDPProblem{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons} 
    # number of constraints
    m::Ti       
    # list of matrices which are sparse/dense/low-rank/diagonal
    constraints::TCons
    # cost matrix
    C::TC
    # right-hand side b
    b::Vector{Tv}
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


mutable struct BurerMonteiro{Tv<:AbstractFloat}
    # primal variables X = RR^T
    R::Matrix{Tv}
    # gradient w.r.t. R
    G::Matrix{Tv}
    # dual variables
    λ::Vector{Tv}
    # violation of constraints
    primal_vio::Vector{Tv}
    # penalty parameter
    σ::Tv
    # objective
    obj::Tv
    # time
    starttime::Tv
    endtime::Tv
    dual_time::Tv
    primal_time::Tv
end

