using SparseArrays
using LinearAlgebra


struct Config 
    ρ_f::Float64
    ρ_c::Float64
    σ_fac::Float64
    #rankreduce::Int
    timelimit::Int
    printlevel::Int
    numlbfgsvecs::Int
    σ_strategy::Int
    λ_updatect::Int
    #rankredtol::Float64
    #gaptol::Float64
    maxmajiter::Int
    maxiter::Int
end


"""
Low rank representation of constraint matrices
written as BDB^T
"""
struct LowRankMatrix{T} <: AbstractMatrix{T}
    D::Diagonal{T, Vector{T}}
    B::Matrix{T}
end


#TODO: support block-wise data
struct ProblemData 
    m::Int # number of constraints      
    # number of constraints of which the matrices are sparse/low-rank/diagonal
    m_sp::Int
    m_diag::Int
    m_lr::Int 
    m_dense::Int
    # list of matrices which are sparse/low-rank/diagonal
    A_sp::Vector{SparseMatrixCSC}
    A_diag::Vector{Diagonal}
    A_lr::Vector{LowRankMatrix}
    A_dense::Vector{Matrix}
    # cost matrix
    C::AbstractMatrix
    # right-hand side b
    b::Vector
end


mutable struct AlgorithmData
    # dual variables
    λ::Vector{Float64}
    # penalty parameter
    σ::Float64
    # objective
    obj::Float64
    # violation of constraints
    vio::Vector{Float64}
    # X = RR^T
    R::Matrix{Float64}
    # gradient
    G::Matrix{Float64}
    # time
    starttime::Float64
end