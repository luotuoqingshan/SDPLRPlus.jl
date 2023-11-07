#using LinearAlgebra, SparseArrays, BenchmarkTools
#using Random, Profile
#include("util.jl")
#
## for reproducing
#Random.seed!(11235813)

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


#function LBFGS_dir!(
#    dir::Matrix{Tv},
#    lbfgshis::LBFGSHistory{Ti, Tv};
#) where {Ti <: Integer, Tv <: AbstractFloat}
#    m = lbfgshis.m
#    lst = lbfgshis.latest[]
#    # pay attention here, dir, s and y are all matrices
#    j = lst
#    for i = 1:m 
#        α = lbfgshis.vecs[j].ρ[] * dot(lbfgshis.vecs[j].s, dir)
#        LinearAlgebra.axpy!(-α, dir, lbfgshis.vecs[j].y)
#        lbfgshis.vecs[j].a[] = α
#        j -= 1
#        if j == 0
#            j = m
#        end
#    end
#
#    j = mod(lst, m) + 1
#    for i = 1:m 
#        β = lbfgshis.vecs[j].ρ[] * dot(lbfgshis.vecs[j].y, dir)
#        γ = lbfgshis.vecs[j].a[] - β
#        LinearAlgebra.axpy!(γ, dir, lbfgshis.vecs[j].s)
#        j += 1
#        if j == m + 1
#            j = 1
#        end
#    end
#end

#numlbfgsvecs = 4 
#n = 8000
#r = 41
#R = randn(n, r)
#dir = randn(n, r)
#lbfgshis = LBFGSHistory{Int64, Float64}(numlbfgsvecs, LBFGSVector{Float64}[], Ref(numlbfgsvecs))
#
#for i = 1:numlbfgsvecs
#    push!(lbfgshis.vecs, 
#        LBFGSVector(similar(R), similar(R), Ref(randn(Float64)), Ref(randn(Float64))))
#end

#@benchmark LBFGS_dir!($dir, $lbfgshis)
#
#Profile.clear()
#@profile LBFGS_dir!(dir, lbfgshis)
#
#save_profile_results(pwd()*"/SDPLR-jl/output/lbfgs_profile_results.txt")
#Profile.clear()
#
#@profile operator2!(dir, R, alpha)
#save_profile_results(pwd()*"/SDPLR-jl/output/operator2_profile_results.txt")
#
#function operator2!(
#    C::Matrix{Tv},
#    A::Matrix{Tv},
#    alpha::Tv,
#) where {Tv <: AbstractFloat}
#    @. C += alpha * A
#end
#
#
#function operator3!(
#    C::Matrix{Tv},
#    A::Matrix{Tv},
#    alpha::Tv,
#) where {Tv <: AbstractFloat}
#    @. C -= alpha * A
#end
#
#alpha = randn(Float64)
#
#@benchmark operator2!($dir, $R, $alpha)
#@benchmark operator3!($dir, $R, $alpha)

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
function dirlbfgs!(
    dir::Matrix{Tv},
    BM::BurerMonteiro{Tv},
    lbfgshis::LBFGSHistory{Ti, Tv};
    negate::Bool=true,
) where{Ti <: Integer, Tv <: AbstractFloat}
    # we store l-bfgs vectors as a cyclic array
    dir .= BM.G
    m = lbfgshis.m
    lst = lbfgshis.latest[]
    # pay attention here, dir, s and y are all matrices
    j = lst
    for i = 1:m 
        α = lbfgshis.vecs[j].ρ[] * dot(lbfgshis.vecs[j].s, dir)
        LinearAlgebra.axpy!(-α, lbfgshis.vecs[j].y, dir)
        lbfgshis.vecs[j].a[] = α
        j -= 1
        if j == 0
            j = m
        end
    end

    j = mod(lst, m) + 1
    for i = 1:m 
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
        rmul!(dir, -one(Tv))
    end

    # partial update of lbfgs history 
    j = mod(lbfgshis.latest[], lbfgshis.m) + 1
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
    j = mod(lbfgshis.latest[], lbfgshis.m) + 1
    @. lbfgshis.vecs[j].s = stepsize * dir
    lbfgshis.vecs[j].y .+= BM.G
    lbfgshis.vecs[j].ρ[] = 1 / dot(lbfgshis.vecs[j].y, lbfgshis.vecs[j].s)
    lbfgshis.latest[] = j
end

