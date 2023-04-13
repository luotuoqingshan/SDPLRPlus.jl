using SparseArrays
using LinearAlgebra

"""
Vector of L-BFGS
"""
struct lbfgsvec
    # notice that we use matrix instead 
    # of vector to store s and y because our
    # decision variables are matrices
    # s = x_{k+1} - x_k
    s::Matrix{Float64}
    # y = \nabla f(x_{k+1}) - \nabla f(x_k)
    y::Matrix{Float64}
    # rho = 1/(y^T s)
    rho::Float64
    # temporary variable
    a::Float64
end


mutable struct lbfgshistory
    m::Int
    vecs::Vector{lbfgsvec}
    latest::Int
end


struct Config 
    rho_f::Float64
    rho_c::Float64
    sigmafac::Float64
    #rankreduce::Int
    timelim::Int
    printlevel::Int
    dthresh_dim::Int
    dthresh_dens::Float64
    numberfgsvecs::Int
    #rankredtol::Float64
    gaptol::Float64
    checkbd::Int
    typebd::Int
end


"""
Low rank representation of constraint matrices
written as BDB^T
"""
struct LowRankMatrix{T} <: AbstractMatrix{T}
    D::Diagonal{Float64}
    B::Matrix{Float64}
end


#TODO: support block-wise data
struct ProblemData 
    m::Int # number of constraints      
    # number of constraints of which the matrices are sparse/low-rank/diagonal
    m_sp::Int
    m_lr::Int 
    m_d::Int
    # list of matrices which are sparse/low-rank/diagonal
    A_sp::Vector{SparseMatrixCSC}
    A_d::Vector{Diagonal}
    A_lr::Vector{LowRankMatrix}
    # right-hand side bs
    C::AbstractMatrix
    bs::Vector
end


mutable struct AlgorithmData
    # dual variables
    lambda::Vector{Float64}
    # penalty parameter
    sigma::Float64
    # violation of constraints
    vio::Vector{Float64}
    # gradient
    G::Matrix{Float64}
    # time
    totaltime::Float64
end