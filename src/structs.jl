"""
Low rank representation of constraint matrices
written as BDBᵀ, since usually B is really thin,
so storing Bᵀ as well doesn't cost too much more storage.
"""
struct LowRankMatrix{T} <: AbstractMatrix{T}
    D::Diagonal{T}
    B::Matrix{T}
    Bt::Matrix{T}
    function LowRankMatrix(
        D::Diagonal{T},
        B::Matrix{T},
    )where {T}
        return new{T}(D, B, Matrix(B'))
    end
end


size(A::LowRankMatrix) = (n = size(A.B, 1); (n, n))
Base.getindex(A::LowRankMatrix, i::Int, j::Int) = (@view(A.Bt[:, i]))' * A.D * @view(A.Bt[:, j])


function show(io::IO, mime::MIME{Symbol("text/plain")}, A::LowRankMatrix)
    summary(io, A) 
    println(io)
    println(io, "LowRankMatrix of form BDBᵀ.")
    println(io, "B factor:")
    show(io, mime, A.B)
    println(io, "\nD factor:")
    show(io, mime, A.D)
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
    Y::AbstractVector{Tv}, 
    A::LowRankMatrix{Tv},
    X::AbstractVector{Tv},
) where {Tv <: AbstractFloat}
    BtX = A.Bt * X
    lmul!(A.D, BtX)
    mul!(Y, A.B, BtX)
end


function LinearAlgebra.mul!(
    Y::AbstractVector{Tv}, 
    A::LowRankMatrix{Tv},
    X::AbstractVector{Tv},
    α::Tv,
    β::Tv,
) where{Tv <: AbstractFloat}
    BtX = A.Bt * X
    lmul!(A.D, BtX)
    mul!(Y, A.B, BtX, α, β)
end


mutable struct SDPProblem{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}} 
    n::Ti                               # size of decision variables
    m::Ti                               # number of constraints
    # list of matrices which are sparse/dense/low-rank/diagonal
    # lowrank_constraints::Vector{LowRankConsOrObj{Ti, Tv}}
    C::TC                               # cost matrix
    b::Vector{Tv}                       # right-hand side b

    # sparse constraints
    XS_colptr::Vector{Ti}
    XS_rowval::Vector{Ti}
    n_sparse_matrices::Ti
    agg_A_ptr::Vector{Ti}
    agg_A_nzind::Vector{Ti}
    agg_A_nzval_one::Vector{Tv}
    agg_A_nzval_two::Vector{Tv}
    sparse_As_global_inds::Vector{Ti}

    X_nzval::Vector{Tv}
    S_nzval::Vector{Tv}
    full_S::SparseMatrixCSC{Tv, Ti}
    full_S_triu_S_inds::Vector{Ti} 
    UVt::Vector{Tv}
    A_RD::Vector{Tv}
    A_DD::Vector{Tv}

    # low-rank constraints
    n_lowrank_matrices::Ti
    lowrank_As::Vector{LowRankMatrix{Tv}}
    lowrank_As_global_inds::Vector{Ti}
    BtVs::Vector{Matrix{Tv}}    # pre-allocated to store Bᵀ * V
    BtUs::Vector{Matrix{Tv}}    # pre-allocated to store Bᵀ * U
    Btvs::Vector{Vector{Tv}}    # pre-allocated to store Bᵀ * v

    R::Matrix{Tv}               # primal variables X = RR^T
    G::Matrix{Tv}               # gradient w.r.t. R
    λ::Vector{Tv}               # dual variables
    y::Vector{Tv}               # auxiliary variable y = -λ + σ * primal_vio
    primal_vio::Vector{Tv}      # violation of constraints

    #scalars::BurerMonterioMutableScalars{Ti, Tv} # mutable scalars

    r::Ti                   # predetermined rank of R, i.e. R ∈ ℝⁿˣʳ
    sigma::Tv               # penalty parameter
    obj::Tv                 # objective
    starttime::Tv           # timing
    endtime::Tv
    dual_time::Tv
    primal_time::Tv
end





