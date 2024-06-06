"""
Vector of L-BFGS
"""
struct LBFGSVector{T}
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
History of L-BFGS vectors
"""
struct LBFGSHistory{Ti <: Integer, Tv}
    # number of l-bfgs vectors
    m::Ti
    vecs::Vector{LBFGSVector{Tv}}
    # the index of the latest l-bfgs vector
    # we use a cyclic array to store l-bfgs vectors
    latest::Base.RefValue{Ti}
end


Base.:length(lbfgshis::LBFGSHistory) = lbfgshis.m


"""
Initialization of L-BFGS history
"""
function lbfgs_init(
    R::Matrix{Tv},
    numlbfgsvecs::Ti,
) where {Ti <: Integer, Tv}
    lbfgsvecs = LBFGSVector{Tv}[]
    for _ = 1:numlbfgsvecs
        push!(lbfgsvecs, 
            LBFGSVector(zeros(Tv, size(R)),
                        zeros(Tv, size(R)),
                        Ref(zero(Tv)), 
                        Ref(zero(Tv)),
                        ))
    end
    lbfgshis = LBFGSHistory{Ti, Tv}(
        numlbfgsvecs,
        lbfgsvecs,
        Ref(numlbfgsvecs))
    return lbfgshis
end

"""
Clear L-BFGS history
"""
function lbfgs_clear!(
    lbfgshis::LBFGSHistory{Ti, Tv}
) where {Ti <: Integer, Tv}
    for i = 1:lbfgshis.m
        lbfgshis.vecs[i].s .= zero(Tv)
        lbfgshis.vecs[i].y .= zero(Tv)
        lbfgshis.vecs[i].ρ[] = zero(Tv)
        lbfgshis.vecs[i].a[] = zero(Tv)
    end
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
) where{Ti <: Integer, Tv}
    # we store l-bfgs vectors as a cyclic array
    copyto!(dir, grad)
    m = lbfgshis.m
    lst = lbfgshis.latest[]

    # if m = 0, LBFGS degenerates to gradient descent 
    if m == 0
        return
    end
    # pay attention here, dir, s and y are all matrices
    j = lst
    for _ = 1:m 
        α = lbfgshis.vecs[j].ρ[] * dot(lbfgshis.vecs[j].s, dir)
        axpy!(-α, lbfgshis.vecs[j].y, dir)
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
    j = mod(lbfgshis.latest[], lbfgshis.m) + 1
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
)where {Ti<:Integer, Tv}
    # if m = 0, LBFGS degenerates to gradient descent
    if lbfgshis.m == 0
        return
    end
    # update lbfgs history
    j = mod(lbfgshis.latest[], lbfgshis.m) + 1

    BLAS.scal!(stepsize, dir)
    copy!(lbfgshis.vecs[j].s, dir)

    axpy!(one(Tv), grad, lbfgshis.vecs[j].y)
    lbfgshis.vecs[j].ρ[] = 1 / dot(lbfgshis.vecs[j].y, lbfgshis.vecs[j].s)

    lbfgshis.latest[] = j
end

