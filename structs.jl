using SparseArrays
using LinearAlgebra
import LinearAlgebra: dot, size, show, norm

"""
Low rank representation of constraint matrices
written as BDB^T
"""
struct LowRankMatrix{T} <: AbstractMatrix{T}
    D::Diagonal{T}
    B::Matrix{T}
end


struct UnitLowRankMatrix{T} <: AbstractMatrix{T}
    B::Matrix{T}
end


size(A::LowRankMatrix) = (n = size(A.B, 1); (n, n))
size(A::UnitLowRankMatrix) = (n = size(A.B, 1); (n, n))


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


function norm(A::LowRankMatrix, p::Real) 
    if p == Inf
        # BDBᵀ 
        U = A.D * A.B'
        s = zero(eltype(U)) 
        @simd for i in axes(U, 2)
            @inbounds s = max(s, norm(A.B * U[:, i], p))
        end
        return s
    elseif p == 2
        U = A.D * A.B'
        s = zero(eltype(U)) 
        @simd for i in axes(U, 2)
            @inbounds s += norm(A.B * U[:, i], p)^2
        end
        return sqrt(s)
    else
        error("undefined norm for LowRankMatrix")
    end
end


function norm(A::UnitLowRankMatrix, p::Real) 
    if p == Inf
        # BDBᵀ 
        U = A.B'
        s = zero(eltype(U)) 
        @simd for i in axes(U, 2)
            @inbounds s = max(s, norm(A.B * U[:, i], p))
        end
        return s
    elseif p == 2
        U = A.B'
        s = zero(eltype(U)) 
        @simd for i in axes(U, 2)
            @inbounds s += norm(A.B * U[:, i], p)^2
        end
        return sqrt(s)
    else
        error("undefined norm for UnitLowRankMatrix")
    end
end


function dot(A::AbstractMatrix{T}, B::LowRankMatrix{T}, C::AbstractMatrix{T}) where T
    if (size(A, 1) != size(B.B, 1) || size(C, 1) != size(B.B, 1)  
        || size(A, 2) != size(C, 2))
        throw(DimensionMismatch("dimension mismatch"))
    end
    U = B.B' * C
    V = B.B' * A
    return dot(V, B.D, U)    
end


function dot(A::AbstractMatrix{T}, B::UnitLowRankMatrix{T}, C::AbstractMatrix{T}) where T
    if (size(A, 1) != size(B.B, 1) || size(C, 1) != size(B.B, 1)  
        || size(A, 2) != size(C, 2))
        throw(DimensionMismatch("dimension mismatch"))
    end
    U = B.B' * C
    V = B.B' * A
    return dot(V, U)    
end


function dot_xTAx(A::LowRankMatrix{T}, X::AbstractMatrix{T}) where T
    if size(X, 1) != size(A.B, 1)
        throw(DimensionMismatch("dimension mismatch"))
    end
    U = A.B' * X
    return dot(U, A.D, U)
end


function dot_xTAx(A::UnitLowRankMatrix{T}, X::AbstractMatrix{T}) where T
    if size(X, 1) != size(A.B, 1)
        throw(DimensionMismatch("dimension mismatch"))
    end
    U = A.B' * X
    return dot(U, U)
end

# fall back function for other matrices
dot_xTAx(A::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T} = dot(X, A, X)

Base.:*(A::AbstractMatrix{T}, B::LowRankMatrix{T}) where {T} = (((A * B.B) * B.D) * B.B')
Base.:*(A::LowRankMatrix{T}, B::AbstractMatrix{T}) where {T} = (A.B * (A.D * (A.B' * B)))
Base.:*(A::AbstractMatrix{T}, B::UnitLowRankMatrix{T}) where {T} = ((A * B.B) * B.B')
Base.:*(A::UnitLowRankMatrix{T}, B::AbstractMatrix{T}) where {T} = (A.B * (A.B' * B))

constraint_eval_UTAU(A::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T} = dot_xTAx(A, X)
constraint_eval_UTAU(A::LowRankMatrix{T}, X::AbstractMatrix{T}) where {T} = dot_xTAx(A, X) 
constraint_eval_UTAU(A::UnitLowRankMatrix{T}, X::AbstractMatrix{T}) where {T} = dot_xTAx(A, X)

constraint_eval_UTAV(A::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T} = (dot(U, A, V) + dot(V, A, U)) / 2
constraint_eval_UTAV(A::LowRankMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T} = (dot(U, A, V) + dot(V, A, U)) / 2
constraint_eval_UTAV(A::UnitLowRankMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T} = (dot(U, A, V) + dot(V, A, U)) / 2

constraint_grad(A::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T} = A * X

#TODO: support block-wise data
struct SDPProblem{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons} 
    # number of constraints
    m::Ti       
    # list of matrices which are sparse/dense/low-rank/diagonal
    constraints::TCons
    #sparse_cons::Vector{SparseMatrixCSC{Tv, Ti}}
    #dense_cons::Vector{Matrix{Tv}}
    #diag_cons::Vector{Diagonal{Tv}}
    #lowrank_cons::Vector{LowRankMatrix{Tv}}
    #unitlowrank_cons::Vector{UnitLowRankMatrix{Tv}}
    # cost matrix
    C::TC
    # right-hand side b
    b::Vector{Tv}
end


function Base.:iterate(SDP::SDPProblem, state=1)
    #base = 0
    #if state <= base + length(SDP.sparse_cons)
    #    return (SDP.sparse_cons[state], state + 1)
    #end
    #base += length(SDP.sparse_cons)
    #if state <= base + length(SDP.dense_cons) 
    #    return (SDP.dense_cons[state - base], state + 1)
    #end
    #base += length(SDP.dense_cons)
    #if state <= base + length(SDP.diag_cons) 
    #    return (SDP.diag_cons[state - base], state + 1)
    #end
    #base += length(SDP.diag_cons)
    #if state <= base + length(SDP.lowrank_cons)
    #    return (SDP.lowrank_cons[state - base], state + 1)
    #end
    #base += length(SDP.lowrank_cons)
    #if state <= base + length(SDP.unitlowrank_cons)
    #    return (SDP.lowrank_cons[state - base], state + 1)
    #end
    #return (nothing, state + 1)
    return state > SDP.m ? nothing : (SDP.constraints[state], state + 1)
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


