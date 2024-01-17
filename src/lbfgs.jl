"""
Vector of L-BFGS
"""
struct LBFGSVector{T <: AbstractFloat}
    # notice that we use matrix instead 
    # of vector to store s and y because our
    # decision variables are matrices
    # s = xₖ₊₁ - xₖ 
    s::Matrix{T}
    # y = ∇ f(xₖ₊₁) - ∇ f(xₖ)
    y::Matrix{T}
    # ρ = 1/(⟨y, s⟩)
    ρ::Base.RefValue{T}
    # temporary variable
    a::Base.RefValue{T}
end

"""
History of l-bfgs vectors
"""
struct LBFGSHistory{Ti <: Integer, Tv <: AbstractFloat}
    # number of l-bfgs vectors
    m::Ti
    vecs::Vector{LBFGSVector{Tv}}
    # the index of the latest l-bfgs vector
    # we use a cyclic array to store l-bfgs vectors
    latest::Base.RefValue{Ti}
end


Base.:length(lbfgshis::LBFGSHistory) = lbfgshis.m


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
function dirlbfgs!(
    dir::Matrix{Tv},
    SDP::SDPProblem{Ti, Tv, TC},
    lbfgshis::LBFGSHistory{Ti, Tv};
    negate::Bool=true,
) where{Ti <: Integer, Tv <: AbstractFloat, TC}
    # we store l-bfgs vectors as a cyclic array
    copyto!(dir, SDP.G)
    m = lbfgshis.m
    lst = lbfgshis.latest[]
    # pay attention here, dir, s and y are all matrices
    j = lst
    for _ = 1:m 
        α = lbfgshis.vecs[j].ρ[] * dot(lbfgshis.vecs[j].s, dir)
        LinearAlgebra.axpy!(-α, lbfgshis.vecs[j].y, dir)
        lbfgshis.vecs[j].a[] = α
        j -= 1
        if j == 0
            j = m
        end
    end

    j = mod(lst, m) + 1
    for _ = 1:m 
        β = lbfgshis.vecs[j].ρ[] * dot(lbfgshis.vecs[j].y, dir)
        γ = lbfgshis.vecs[j].a[] - β
        LinearAlgebra.axpy!(γ, lbfgshis.vecs[j].s, dir)
        j += 1
        if j == m + 1
            j = 1
        end
    end

    # we need to pick -dir as search direction
    if negate 
        LinearAlgebra.BLAS.scal!(-one(Tv), dir)
    end

    # partial update of lbfgs history 
    j = mod(lbfgshis.latest[], lbfgshis.m) + 1
    copyto!(lbfgshis.vecs[j].y, SDP.G)
    LinearAlgebra.BLAS.scal!(-one(Tv), lbfgshis.vecs[j].y)
end


"""
Postprocessing step of L-BFGS.
"""
function lbfgs_postprocess!(
    SDP::SDPProblem{Ti, Tv, TC},
    lbfgshis::LBFGSHistory{Ti, Tv},
    dir::Matrix{Tv},
    stepsize::Tv,
)where {Ti<:Integer, Tv <: AbstractFloat, TC}
    # update lbfgs history
    j = mod(lbfgshis.latest[], lbfgshis.m) + 1

    LinearAlgebra.BLAS.scal!(stepsize, dir)
    copy!(lbfgshis.vecs[j].s, dir)

    LinearAlgebra.axpy!(one(Tv), SDP.G, lbfgshis.vecs[j].y)
    lbfgshis.vecs[j].ρ[] = 1 / dot(lbfgshis.vecs[j].y, lbfgshis.vecs[j].s)

    lbfgshis.latest[] = j
end

