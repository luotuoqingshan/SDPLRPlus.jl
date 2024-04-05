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


LinearAlgebra.size(A::SymLowRankMatrix) = (n = size(A.B, 1); (n, n))
Base.getindex(A::SymLowRankMatrix, i::Integer, j::Integer) = (@view(A.Bt[:, i]))' * A.D * @view(A.Bt[:, j])


function LinearAlgebra.show(io::IO, mime::MIME{Symbol("text/plain")}, A::SymLowRankMatrix)
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
function LinearAlgebra.norm(
    A::SymLowRankMatrix{Tv},
    p::Real,
) where {Tv <: Number}
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

Multiply a symmetric low-rank matrix `A` of the form `BDBᵀ` with an AbstractArray `X`.
"""
function LinearAlgebra.mul!(
    Y::AbstractVecOrMat{Tv}, 
    A::SymLowRankMatrix{Tv},
    X::AbstractVecOrMat{Tv},
) where {Tv <: Number}
    BtX = A.Bt * X
    lmul!(A.D, BtX)
    mul!(Y, A.B, BtX)
end

function LinearAlgebra.mul!(
    Y::AbstractMatrix{Tv}, 
    X::AbstractVecOrMat{Tv},
    A::SymLowRankMatrix{Tv},
) where {Tv <: Number}
    XB = X * A.B 
    rmul!(XB, A.D)
    mul!(Y, XB, A.Bt)
end


"""
    mul!(Y, A, X, α, β)

Compute `Y = α * A * X + β * Y` where `A` is a symmetric low-rank matrix of the form `BDBᵀ`.
"""
function LinearAlgebra.mul!(
    Y::AbstractVecOrMat{Tv}, 
    A::SymLowRankMatrix{Tv},
    X::AbstractVecOrMat{Tv},
    α::Tv,
    β::Tv,
) where{Tv <: Number}
    BtX = A.Bt * X
    lmul!(A.D, BtX)
    mul!(Y, A.B, BtX, α, β)
end


function LinearAlgebra.mul!(
    Y::AbstractMatrix{Tv}, 
    X::AbstractVecOrMat{Tv},
    A::SymLowRankMatrix{Tv},
    α::Tv,
    β::Tv,
) where{Tv <: Number}
    XB = X * A.B 
    rmul!(XB, A.D)
    mul!(Y, XB, A.Bt, α, β)
end


struct SDPData{Ti <: Integer, Tv, TC <: AbstractMatrix{Tv}}
    n::Ti                               # size of decision variables
    m::Ti                               # number of constraints
    C::TC                               # cost matrix
    As::Vector{Any}                     # set of constraint matrices
    b::Vector{Tv}                       # right-hand side b
end


struct SolverVars{Ti <: Integer,Tv}
    Rt::Matrix{Tv}               # primal variables X = RR^T
    Gt::Matrix{Tv}               # gradient w.r.t. R
    λ::Vector{Tv}               # dual variables

    r::Base.RefValue{Ti}                       # predetermined rank of R, i.e. R ∈ ℝⁿˣʳ
    σ::Base.RefValue{Tv}                       # penalty parameter
    obj::Base.RefValue{Tv}                     # objective
end


struct SolverAuxiliary{Ti <: Integer, Tv}
    # sparse constraints
    n_sparse_matrices::Ti
    triu_agg_sparse_A_matptr::Vector{Ti}
    triu_agg_sparse_A_nzind::Vector{Ti}
    triu_agg_sparse_A_nzval_one::Vector{Tv}
    triu_agg_sparse_A_nzval_two::Vector{Tv}
    agg_sparse_A_mappedto_triu::Vector{Ti} 
    sparse_As_global_inds::Vector{Ti}

    triu_sparse_S::SparseMatrixCSC{Tv, Ti}
    sparse_S::SparseMatrixCSC{Tv, Ti}
    UVt::Vector{Tv}
    A_RD::Vector{Tv}
    A_DD::Vector{Tv}
    
    # symmetric low-rank constraints
    n_symlowrank_matrices::Ti
    symlowrank_As::Vector{SymLowRankMatrix{Tv}}
    symlowrank_As_global_inds::Vector{Ti}

    y::Vector{Tv}               # auxiliary variable y = -λ + σ * primal_vio
    primal_vio::Vector{Tv}      # violation of constraints
end


struct SolverStats{Ti <: Integer, Tv}
    starttime::Base.RefValue{Tv}               # timing
    endtime::Base.RefValue{Tv}
    dual_lanczos_time::Base.RefValue{Tv}
    dual_GenericArpack_time::Base.RefValue{Tv}
    checkdualbd_iters::Vector{Ti}
    lanczos_eigvals::Vector{Tv}
    GenericArpack_eigvals::Vector{Tv}
    primal_time::Base.RefValue{Tv}
    DIMACS_time::Base.RefValue{Tv}  
end




