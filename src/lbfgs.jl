"""
Vector of L-BFGS
"""
mutable struct LBFGSVector{T <: AbstractFloat}
    # notice that we use matrix instead 
    # of vector to store s and y because our
    # decision variables are matrices
    # s = xₖ₊₁ - xₖ 
    s::Matrix{T}
    # y = ∇ f(xₖ₊₁) - ∇ f(xₖ)
    y::Matrix{T}
    # ρ = 1/(⟨y, s⟩)
    #ρ::Base.RefValue{T}
    ρ::T
    # temporary variable
    #a::Base.RefValue{T}
    a::T
end

"""
History of L-BFGS vectors
"""
mutable struct LBFGSHistory{Ti <: Integer, Tv <: AbstractFloat}
    # number of l-bfgs vectors
    m::Ti
    vecs::Vector{LBFGSVector{Tv}}
    # the index of the latest l-bfgs vector
    # we use a cyclic array to store l-bfgs vectors
    #latest::Base.RefValue{Ti}
    latest::Ti
end


Base.:length(lbfgshis::LBFGSHistory) = lbfgshis.m


"""
Initialization of L-BFGS history
"""
function lbfgs_init(
    R::Matrix{Tv},
    numlbfgsvecs::Ti,
) where {Ti <: Integer, Tv <: AbstractFloat}
    lbfgshis = LBFGSHistory{Ti, Tv}(
        numlbfgsvecs,
        LBFGSVector{Tv}[],
        numlbfgsvecs)

    for _ = 1:numlbfgsvecs
        push!(lbfgshis.vecs, 
            LBFGSVector(zeros(Tv, size(R)),
                        zeros(Tv, size(R)),
                        zero(Tv), 
                        zero(Tv),
                        ))
    end

    return lbfgshis
end


"""
L-BFGS two-loop recursion
to compute the direction

q = ∇ fₖ
for i = k - 1, k - 2, ... k - m
    αᵢ = ρᵢ * sᵢᵀ q 
    q = q - αᵢ * yᵢ 
end
r = Hₖ * q # we don't do this step

for i = k - m, k - m + 1, ... k - 1
    βᵢ = ρᵢ * yᵢᵀ r
    r = r + sᵢ * (αᵢ - βᵢ)
end
"""
function lbfgs_dir!(
    dir::Matrix{Tv},
    lbfgshis::LBFGSHistory{Ti, Tv},
    grad::Matrix{Tv};
    negate::Bool=true,
) where{Ti <: Integer, Tv <: AbstractFloat}
    # we store l-bfgs vectors as a cyclic array
    copyto!(dir, grad)
    m = lbfgshis.m
    lst = lbfgshis.latest
    # pay attention here, dir, s and y are all matrices
    j = lst
    for _ = 1:m 
        α = lbfgshis.vecs[j].ρ * dot(lbfgshis.vecs[j].s, dir)
        axpy!(-α, lbfgshis.vecs[j].y, dir)
        lbfgshis.vecs[j].a = α
        j -= 1
        if j == 0
            j = m
        end
    end

    j = mod(lst, m) + 1
    for _ = 1:m 
        β = lbfgshis.vecs[j].ρ * dot(lbfgshis.vecs[j].y, dir)
        γ = lbfgshis.vecs[j].a - β
        axpy!(γ, lbfgshis.vecs[j].s, dir)
        j += 1
        if j == m + 1
            j = 1
        end
    end

    # we need to pick -dir as search direction
    if negate 
        BLAS.scal!(-one(Tv), dir)
    end

    # partial update of lbfgs history 
    j = mod(lbfgshis.latest, lbfgshis.m) + 1
    copyto!(lbfgshis.vecs[j].y, grad)
    BLAS.scal!(-one(Tv), lbfgshis.vecs[j].y)
end


"""
Postprocessing step of L-BFGS.
"""
function lbfgs_update!(
    dir::Matrix{Tv},
    lbfgshis::LBFGSHistory{Ti, Tv},
    grad::Matrix{Tv},
    stepsize::Tv,
)where {Ti<:Integer, Tv <: AbstractFloat}
    # update lbfgs history
    j = mod(lbfgshis.latest, lbfgshis.m) + 1

    BLAS.scal!(stepsize, dir)
    copy!(lbfgshis.vecs[j].s, dir)

    axpy!(one(Tv), grad, lbfgshis.vecs[j].y)
    lbfgshis.vecs[j].ρ, = 1 / dot(lbfgshis.vecs[j].y, lbfgshis.vecs[j].s)

    lbfgshis.latest = j
end

