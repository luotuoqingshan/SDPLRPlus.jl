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
    ρ::T
    # temporary variable
    a::T
end

"""
History of l-bfgs vectors
"""
mutable struct LBFGSHistory{Ti <: Integer, Tv <: AbstractFloat}
    # number of l-bfgs vectors
    m::Ti
    vecs::Vector{LBFGSVector{Tv}}
    # the index of the latest l-bfgs vector
    # we use a cyclic array to store l-bfgs vectors
    latest::Ti
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

Notice here if we initialize all sᵢ, yᵢ to be zero
then we don't need to record how many
sᵢ, yᵢ pairs we have already computed 
"""
function dirlbfgs(
    BM::BurerMonteiro{Tv},
    lbfgshis::LBFGSHistory{Ti, Tv};
    negate::Bool=true,
) where{Ti <: Integer, Tv <: AbstractFloat}
    # we store l-bfgs vectors as a cyclic array
    dir = copy(BM.G)
    m = lbfgshis.m
    lst = lbfgshis.latest
    # pay attention here, dir, s and y are all matrices
    j = lst
    for i = 1:m 
        α = lbfgshis.vecs[j].ρ * dot(lbfgshis.vecs[j].s, dir)
        @. dir -= lbfgshis.vecs[j].y * α 
        lbfgshis.vecs[j].a = α
        j -= 1
        if j == 0
            j = m
        end
    end

    j = mod(lst, m) + 1
    for i = m:1
        β = lbfgshis.vecs[j].ρ * dot(lbfgshis.vecs[j].y, dir)
        @. dir += lbfgshis.vecs[j].s * (lbfghis.vec[i].a - β) 
    end

    # we need to pick -dir as search direction
    if negate 
        rmul!(dir, -one(Tv))
    end

    # partial update of lbfgs history 
    j = mod(lbfgshis.latest, lbfgshis.m) + 1
    lbfgshis.vecs[j].y .= -BM.G
    return dir
end


function lbfgs_postprocess!(
    BM::BurerMonteiro{T},
    lbfgshis::LBFGSHistory{<:Integer, T},
    dir::Matrix{T},
    stepsize::T,
)where T
    # update lbfgs history
    j = mod(lbfgshis.latest, lbfgshis.m) + 1
    @. lbfgshis.vecs[j].s = stepsize * dir
    lbfgshis.vecs[j].y .+= BM.G
    lbfgshis.vecs[j].ρ = 1 / dot(lbfgshis.vecs[j].y, lbfgshis.vecs[j].s)
    lbfgshis.latest = j
end


