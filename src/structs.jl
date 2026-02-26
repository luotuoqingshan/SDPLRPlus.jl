@doc raw"""
    SymLowRankMatrix{T} <: AbstractMatrix{T}

Symmetric low-rank matrix of the form ``BDB^T`` with elements of type `T`.
Besides the diagonal matrix ``D`` and the thin matrix ``B``,
we also store ``B^T`` as `Bt`.
It's because usually `B` is really thin, 
storing `Bt` doesn't cost too much more storage
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
    function SymLowRankMatrix(D::Diagonal{T}, B::Matrix{T}) where {T}
        return new{T}(D, B, Matrix(B'))
    end
end

"""
size of a symmetric low-rank matrix
"""
LinearAlgebra.size(A::SymLowRankMatrix) = (n = size(A.B, 1); (n, n))

"""
    getindex(A, i, j)

return the (i, j)-th element of a symmetric low-rank matrix `A` of the form `BDBᵀ`.
"""
(function Base.getindex(A::SymLowRankMatrix, i::Integer, j::Integer)
    return (@view(A.Bt[:, i]))' * A.D * @view(A.Bt[:, j])
end)

"""
display a symmetric low-rank matrix
"""
function LinearAlgebra.show(
    io::IO, mime::MIME{Symbol("text/plain")}, A::SymLowRankMatrix
)
    summary(io, A)
    println(io)
    println(io, "SymLowRankMatrix of form BDBᵀ.")
    println(io, "B factor:")
    show(io, mime, A.B)
    println(io, "\nD factor:")
    return show(io, mime, A.D)
end

"""
    norm(A, p)

Compute the `p`-norm of a symmetric low-rank matrix `A` of the form `BDBᵀ`.
Currently support `p` ∈ [2, Inf]. 
"""
function LinearAlgebra.norm(A::SymLowRankMatrix{Tv}, p::Real) where {Tv<:Number}
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

Multiply a symmetric low-rank matrix `A` of the form `BDBᵀ` 
with an AbstractArray `X`.
"""
function LinearAlgebra.mul!(
    Y::AbstractVecOrMat{Tv}, A::SymLowRankMatrix{Tv}, X::AbstractVecOrMat{Tv}
) where {Tv<:Number}
    BtX = A.Bt * X
    lmul!(A.D, BtX)
    return mul!(Y, A.B, BtX)
end

"""
    mul!(Y, X, A)

Multiply an AbstractArray `X` with a symmetric low-rank matrix `A` of the form `BDBᵀ`.
"""
function LinearAlgebra.mul!(
    Y::AbstractMatrix{Tv}, X::AbstractVecOrMat{Tv}, A::SymLowRankMatrix{Tv}
) where {Tv<:Number}
    XB = X * A.B
    rmul!(XB, A.D)
    return mul!(Y, XB, A.Bt)
end

"""
    mul!(Y, A, X, α, β)

Compute `Y = α * A * X + β * Y` 
where `A` is a symmetric low-rank matrix of the form `BDBᵀ`.
"""
function LinearAlgebra.mul!(
    Y::AbstractVecOrMat{Tv},
    A::SymLowRankMatrix{Tv},
    X::AbstractVecOrMat{Tv},
    α::Tv,
    β::Tv,
) where {Tv<:Number}
    BtX = A.Bt * X
    lmul!(A.D, BtX)
    return mul!(Y, A.B, BtX, α, β)
end

"""
    mul!(Y, X, A, α, β)

Compute `Y = α * X * A + β * Y` 
where `A` is a symmetric low-rank matrix of the form `BDBᵀ`.
"""
function LinearAlgebra.mul!(
    Y::AbstractMatrix{Tv},
    X::AbstractVecOrMat{Tv},
    A::SymLowRankMatrix{Tv},
    α::Tv,
    β::Tv,
) where {Tv<:Number}
    XB = X * A.B
    rmul!(XB, A.D)
    return mul!(Y, XB, A.Bt, α, β)
end

"""
Structure for storing the data of a semidefinite programming problem.
"""
struct SDPData{Ti<:Integer,Tv,TC<:AbstractMatrix{Tv},TA}
    n::Ti                               # size of decision variables
    m::Ti                               # number of constraints
    C::TC                               # cost matrix
    As::Vector{TA}                      # set of constraint matrices
    b::Vector{Tv}                       # right-hand side b
    # constraint type: false = equality (𝒜ᵢ(X) = bᵢ)
    #                  true  = inequality (𝒜ᵢ(X) ≤ bᵢ)
    constraint_types::Vector{Bool}
    has_inequalities::Bool              # true iff any(constraint_types)
end

# backward-compat: all equality
function SDPData(C, As, b)
    return SDPData(
        size(C, 1), length(As), C, As, b, fill(false, length(As)), false
    )
end

# with explicit constraint types
function SDPData(C, As, b, constraint_types)
    return SDPData(
        size(C, 1),
        length(As),
        C,
        As,
        b,
        constraint_types,
        any(constraint_types),
    )
end

b_vector(model) = model.b
C_matrix(model) = model.C

# The scalar variables in the following three structures  
# are stored by RefValue such that we can declare the structures 
# as unmutable structs.
# see discussion here:
# https://discourse.julialang.org/t/question-on-refvalue/53498/2

"""
Structure for storing the variables used in the solver.
"""
struct SolverVars{Ti<:Integer,Tv,TR<:AbstractArray{Tv}}
    Rt::TR              # primal variables X = RR^T
    Gt::TR              # gradient w.r.t. R
    λ::Vector{Tv}               # dual variables
    # upper bound for dual variables
    # 0 for dual variables for inequalities 
    # and Inf for dual variables for equalities
    λ_ub::Vector{Tv}

    r::Base.RefValue{Ti}        # predetermined rank of R, i.e. R ∈ ℝⁿˣʳ
    σ::Base.RefValue{Tv}        # penalty parameter
    obj::Base.RefValue{Tv}      # objective

    # auxiliary variable y = -λ + σ * primal_vio_raw
    y::Vector{Tv}
    # raw violation of constraints, for convenience, we store
    # a length (m+1) vector where
    # the first m entries correspond to the primal violation
    # and the last entry corresponds to the objective
    primal_vio_raw::Vector{Tv}
    # lower bound for capping primal violations (length m):
    #   equality:   -Inf  (vᵢ uncapped — both signs count)
    #   inequality:  0.0  (only positive violations count)
    primal_vio_lb::Vector{Tv}
    # capped violation: primal_vio[i] = max(primal_vio_raw[i], primal_vio_lb[i])
    # norm(primal_vio, p) gives the correct feasibility measure for any p
    primal_vio::Vector{Tv}
    A_RD::Vector{Tv}
    A_DD::Vector{Tv}
end

function SolverVars(
    data::SDPData{Ti,Tv}, r, config::BurerMonteiroConfig
) where {Ti,Tv}
    # derive λ upper bounds from constraint types:
    # equality → Inf (unrestricted), inequality ≤ → 0 (λ ≤ 0)
    λ_ub = [ct ? zero(Tv) : Tv(Inf) for ct in data.constraint_types]
    if config.init_func !== nothing
        Rt0, λ0 = config.init_func(data, r, config.init_args...)
        @. λ0 = min(λ0, λ_ub)
        return SolverVars(Rt0, λ0, λ_ub, r, config.σ_0)
    else
        Rt0 = 2 .* rand(r, data.n) .- 1
        λ0 = zeros(Tv, data.m)
        return SolverVars(Rt0, λ0, λ_ub, r, config.σ_0)
    end
end

function SolverVars(
    Rt0, λ0::Vector{Tv}, λ_ub::Vector{Tv}, r, σ_0::Tv=2.0
) where {Tv}
    m = length(λ0)
    # equality (λ_ub=Inf): lb=-Inf; inequality (λ_ub=0): lb=0
    primal_vio_lb = [isinf(ub) ? Tv(-Inf) : zero(Tv) for ub in λ_ub]
    return SolverVars(
        Rt0,
        zeros(Tv, size(Rt0)),
        λ0,
        λ_ub,
        Ref(r),
        Ref(σ_0),
        Ref(zero(Tv)),
        zeros(Tv, m + 1), # y
        zeros(Tv, m + 1), # primal_vio_raw
        primal_vio_lb,
        zeros(Tv, m),     # primal_vio
        zeros(Tv, m + 1), # A_RD
        zeros(Tv, m + 1), # A_DD
    )
end

# backward-compat: all-equality (λ_ub = Inf everywhere)
function SolverVars(Rt0, λ0::Vector{Tv}, r, σ_0::Tv=2.0) where {Tv}
    return SolverVars(Rt0, λ0, fill(Tv(Inf), length(λ0)), r, σ_0)
end

"""
Structure for auxiliary variables used in the solver.
In General, user should not directly interact with this structure.
"""
struct SolverAuxiliary{Ti<:Integer,Tv}
    # auxiliary variables for sparse constraints
    # take a look at the preprocess_sparsecons function
    # for more explanation
    n_sparse_matrices::Ti
    triu_agg_sparse_A_matptr::Vector{Ti}
    triu_agg_sparse_A_nzind::Vector{Ti}
    triu_agg_sparse_A_nzval_one::Vector{Tv}
    triu_agg_sparse_A_nzval_two::Vector{Tv}
    agg_sparse_A_mappedto_triu::Vector{Ti}
    sparse_As_global_inds::Vector{Ti}

    triu_sparse_S::SparseMatrixCSC{Tv,Ti}
    sparse_S::SparseMatrixCSC{Tv,Ti}
    UVt::Vector{Tv}

    # symmetric low-rank constraints
    n_symlowrank_matrices::Ti
    symlowrank_As::Vector{SymLowRankMatrix{Tv}}
    symlowrank_As_global_inds::Vector{Ti}
end

function SolverAuxiliary(data::SDPData{Ti,Tv}) where {Ti,Tv}
    sparse_cons = Union{SparseMatrixCSC{Tv,Ti},SparseMatrixCOO{Tv,Ti}}[]
    symlowrank_cons = SymLowRankMatrix{Tv}[]
    # treat diagonal matrices as sparse matrices
    sparse_As_global_inds = Ti[]
    symlowrank_As_global_inds = Ti[]

    for (i, A) in enumerate(data.As)
        if isa(A, Union{SparseMatrixCSC,SparseMatrixCOO})
            push!(sparse_cons, A)
            push!(sparse_As_global_inds, i)
        elseif isa(A, Diagonal)
            push!(sparse_cons, sparse(A))
            push!(sparse_As_global_inds, i)
        elseif isa(A, SymLowRankMatrix)
            push!(symlowrank_cons, A)
            push!(symlowrank_As_global_inds, i)
        else
            @error "Currently only sparse\
            /symmetric low-rank\
            /diagonal constraints are supported."
        end
    end

    if isa(data.C, Union{SparseMatrixCSC,SparseMatrixCOO})
        push!(sparse_cons, data.C)
        push!(sparse_As_global_inds, data.m + 1)
    elseif isa(data.C, Diagonal)
        push!(sparse_cons, sparse(data.C))
        push!(sparse_As_global_inds, data.m + 1)
    elseif isa(data.C, SymLowRankMatrix)
        push!(symlowrank_cons, data.C)
        push!(symlowrank_As_global_inds, data.m + 1)
    else
        @error "Currently only sparse\
        /lowrank/diagonal objectives are supported."
    end

    @info "Finish classifying constraints."

    # preprocess sparse constraints
    res = @timed begin
        triu_agg_sparse_A, triu_agg_sparse_A_matptr, triu_agg_sparse_A_nzind, triu_agg_sparse_A_nzval_one, triu_agg_sparse_A_nzval_two, agg_sparse_A, agg_sparse_A_mappedto_triu = preprocess_sparsecons(
            sparse_cons
        )
    end
    @debug "$(res.bytes)B allocated during preprocessing constraints."

    nnz_triu_agg_sparse_A = length(triu_agg_sparse_A.rowval)

    return SolverAuxiliary(
        length(sparse_cons),
        triu_agg_sparse_A_matptr,
        triu_agg_sparse_A_nzind,
        triu_agg_sparse_A_nzval_one,
        triu_agg_sparse_A_nzval_two,
        agg_sparse_A_mappedto_triu,
        sparse_As_global_inds,
        triu_agg_sparse_A,
        agg_sparse_A,
        zeros(Tv, nnz_triu_agg_sparse_A), # UVt
        length(symlowrank_cons), #n_symlowrank_matrices
        symlowrank_cons,
        symlowrank_As_global_inds,
    )
end

side_dimension(aux::SolverAuxiliary) = size(aux.sparse_S, 1)

struct SolverStats{Tv}
    starttime::Base.RefValue{Tv}               # timing
    endtime::Base.RefValue{Tv}                 # time spent on computing eigenvalue using Lanczos with random start
    dual_lanczos_time::Base.RefValue{Tv}       # time spent on computing eigenvalue using GenericArpack 
    dual_GenericArpack_time::Base.RefValue{Tv} # total time - dual_lanczos_time - dual_GenericArpack_time 
    primal_time::Base.RefValue{Tv}
    DIMACS_time::Base.RefValue{Tv}  # time spent on computing the DIMACS stats which is not included in the total time
    function SolverStats{Tv}() where {Tv}
        return new{Tv}(
            Ref(zero(Tv)), # starttime
            Ref(zero(Tv)), # endtime
            Ref(zero(Tv)), # time spent on lanczos with random start
            Ref(zero(Tv)), # time spent on GenericArpack
            Ref(zero(Tv)), # primal time
            Ref(zero(Tv)), # DIMACS time
        )
    end
end
