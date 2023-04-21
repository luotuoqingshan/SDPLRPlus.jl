"""
Vector of L-BFGS
"""
mutable struct lbfgsvec
    # notice that we use matrix instead 
    # of vector to store s and y because our
    # decision variables are matrices
    # s = xₖ₊₁ - xₖ 
    s::Matrix{Float64}
    # y = ∇ f(xₖ₊₁) - ∇ f(xₖ)
    y::Matrix{Float64}
    # ρ = 1/(Tr(yᵀs))
    ρ::Float64
    # temporary variable
    a::Float64
end

"""
History of l-bfgs vectors
"""
mutable struct lbfgshistory
    # number of l-bfgs vectors
    m::Int
    vecs::Vector{lbfgsvec}
    # the index of the latest l-bfgs vector
    # we use a cyclic array to store l-bfgs vectors
    latest::Int
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

Notice here if we initialize all sᵢ, yᵢ to be zero
then we don't need to record how many
sᵢ, yᵢ pairs we have already computed 
"""
function dirlbfgs(
    algdata::AlgorithmData,
    lbfgshis::lbfgshistory;
    negate::Bool=true,
)
    # we store l-bfgs vectors as a cyclic array
    dir = algdata.G
    m = lbfgshis.m
    lst = lbfgshis.latest
    # pay attention here, dir, s and y are all matrices
    for i = 1:m
        j = mod(lst - i + m, m) + 1 # j = (lst - (i-1)) 
        α = sum(lbfgshis.vecs[j].ρ * (lbfgshis.vecs[j].s .* dir))
        dir -= lbfgshis.vecs[j].y * α 
        lbfgshis.vecs[j].a = α
    end

    for i = m:1
        j = mod(lst - i + m, m) + 1 
        β = lbfgshis.vecs[j].ρ * (lbfgshis.vecs[j].y .* dir)
        dir += lbfgshis.vecs[j].s * (lbfghis.vec[i].a - β) 
    end

    # we need to pick -dir as search direction
    if negate 
        dir .*= -1
    end

    # partial update of lbfgs history 
    j = (lbfgshis.latest) % lbfgshis.m + 1
    lbfgshis.vecs[j].y = -algdata.G
    return dir
end


function lbfgs_postprocess!(
    algdata::AlgorithmData,
    lbfgshis::lbfgshistory,
    dir::Matrix{Float64},
    stepsize::Float64,
)
    # update lbfgs history
    j = (lbfgshis.latest) % lbfgshis.m + 1
    lbfgshis.vecs[j].s = stepsize * dir
    lbfgshis.vecs[j].y += algdata.G
    lbfgshis.vecs[j].ρ = 1 / sum(lbfgshis.vecs[j].y .* lbfgshis.vecs[j].s)
    lbfgshis.latest = j
end

