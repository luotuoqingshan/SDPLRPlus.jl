"""
    SymLowRankMatrix{T}

Symmetric low-rank matrix of the form BDBᵀ with elements of type `T`.
Besides the diagonal matrix `D` and the thin matrix `B`,
we also store the transpose of `B` as `Bt`.
It's because usually `B` is really thin, storing `Bt` doesn't cost too much more storage
but will save allocation during computation.
"""
struct SymLowRankMatrix{T} <: AbstractMatrix{T}
    D::Diagonal{T}
    B::Matrix{T}
    Bt::Matrix{T}

    @doc """
        SymLowRankMatrix(D, B)
    
    Construct a symmetric low-rank matrix of the form `BDBᵀ`.
    """
    function SymLowRankMatrix(
        D::Diagonal{T},
        B::Matrix{T},
    )where {T}
        return new{T}(D, B, Matrix(B'))
    end
end


size(A::SymLowRankMatrix) = (n = size(A.B, 1); (n, n))
Base.getindex(A::SymLowRankMatrix, i::Integer, j::Integer) = (@view(A.Bt[:, i]))' * A.D * @view(A.Bt[:, j])


function show(io::IO, mime::MIME{Symbol("text/plain")}, A::SymLowRankMatrix)
    summary(io, A) 
    println(io)
    println(io, "SymLowRankMatrix of form BDBᵀ.")
    println(io, "B factor:")
    show(io, mime, A.B)
    println(io, "\nD factor:")
    show(io, mime, A.D)
end


"""
    norm(A, p)

Compute the `p`-norm of a symmetric low-rank matrix `A` of the form `BDBᵀ`.
Currently support `p` ∈ [2, Inf]. 
"""
function norm(
    A::SymLowRankMatrix{Tv},
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


"""
    mul!(Y, A, X)

Multiply a symmetric low-rank matrix `A` of the form `BDBᵀ` with an AbstractVector `X`.
"""
function LinearAlgebra.mul!(
    Y::AbstractVector{Tv}, 
    A::SymLowRankMatrix{Tv},
    X::AbstractVector{Tv},
) where {Tv <: AbstractFloat}
    BtX = A.Bt * X
    lmul!(A.D, BtX)
    mul!(Y, A.B, BtX)
end


"""
    mul!(Y, A, X, α, β)

Compute `Y = α * A * X + β * Y` where `A` is a symmetric low-rank matrix of the form `BDBᵀ`.
"""
function LinearAlgebra.mul!(
    Y::AbstractVector{Tv}, 
    A::SymLowRankMatrix{Tv},
    X::AbstractVector{Tv},
    α::Tv,
    β::Tv,
) where{Tv <: AbstractFloat}
    BtX = A.Bt * X
    lmul!(A.D, BtX)
    mul!(Y, A.B, BtX, α, β)
end


"""
    SDPProblem
"""
mutable struct SDPProblem{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}} 
    n::Ti                               # size of decision variables
    m::Ti                               # number of constraints
    # list of matrices which are sparse/dense/low-rank/diagonal
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

    # symmetric low-rank constraints
    n_symlowrank_matrices::Ti
    symlowrank_As::Vector{SymLowRankMatrix{Tv}}
    symlowrank_As_global_inds::Vector{Ti}
    BtVs::Vector{Matrix{Tv}}    # pre-allocated to store Bᵀ * V
    BtUs::Vector{Matrix{Tv}}    # pre-allocated to store Bᵀ * U
    Btvs::Vector{Vector{Tv}}    # pre-allocated to store Bᵀ * v

    R::Matrix{Tv}               # primal variables X = RR^T
    G::Matrix{Tv}               # gradient w.r.t. R
    λ::Vector{Tv}               # dual variables
    y::Vector{Tv}               # auxiliary variable y = -λ + σ * primal_vio
    primal_vio::Vector{Tv}      # violation of constraints

    r::Ti                   # predetermined rank of R, i.e. R ∈ ℝⁿˣʳ
    sigma::Tv               # penalty parameter
    obj::Tv                 # objective
    starttime::Tv           # timing
    endtime::Tv
    dual_time::Tv
    primal_time::Tv
end





