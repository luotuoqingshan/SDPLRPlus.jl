include("header.jl")
using BenchmarkTools

#BLAS.set_num_threads(4)
Random.seed!(0)
graph = "G67"
A = load_gset(graph) 
#A = sparse([0 1; 1 0])

C, As, bs = maxcut(A)
@show nnz(C)
n = size(A, 1)
r = 20 #barvinok_pataki(n, n)  
size(A)
nnz(A)

res = sdplr(C, As, bs, r)

#using SparseArrays, LinearAlgebra, MKLSparse
#n = 100000
#r = 30
#p = 0.001
#A = sprand(n, n, p)
#A = max.(A, A')
#B = randn(n, r)
#Bt = Matrix(B')
#
#BLAS.set_num_threads(4)
#using BenchmarkTools
#@benchmark $A*$B
#@benchmark $Bt*$A








#r = 5
#using Profile
#@profile sdplr(C, As, bs, r)
#save_profile_results(pwd()*"/SDPLR-jl/output/Nov-7-6:47pm-profile-result.txt")
#
#BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
#BenchmarkTools.DEFAULT_PARAMETERS.evals = 5
#
#benchmark_res = @benchmark sdplr($C, $As, $bs, $r)
#filepath = pwd()*"/SDPLR-jl/output/Nov-6-6:22pm-G11.txt"
#
#include("util.jl")
#save_benchmark_results(benchmark_res, filepath)
#
#
#@profile sdplr(C, As, bs, r)
#
#cmd = `sdplr data/G11.sdpa sdplr.params data/G11.solin`
#c_dt = @elapsed run(cmd) 
#
#solinpath = pwd()*"/SDPLR-jl/data/Gset/G11.minbisec.solin"
#sdpapath = pwd()*"/SDPLR-jl/data/Gset/G11.minbisec.sdpa"
#
#write_initial_solution(res["R₀"], res["λ₀"], solinpath)
#write_problem_sdpa(C, As, bs, sdpapath)
#
#
#
##using SparseArrays, LinearAlgebra, BenchmarkTools
##function op1(A::SparseMatrixCSC{T}, R::Matrix{T}) where{T <: AbstractFloat}
##    return sum(R .* (A * R))
##end
##
##function op2(A::SparseMatrixCSC{T}, R::Matrix{T}) where{T <: AbstractFloat}
##    return dot(R, A, R) 
##end
##
##function op3(A::SparseMatrixCSC{T}, R::Matrix{T}) where{T <: AbstractFloat}
##    Rt = R'
##    res = zero(T) 
##    @inbounds for (x, y, v) in zip(findnz(A)...)
##        res += v * dot(@view(Rt[:, x]), @view(Rt[:, y]))
##    end
##    return res
##end
##n = 800
##r = 41
##R = randn(n, r)
##A = sprand(n, n, 0.01)
##@btime op1($A, $R)
##@btime op2($A, $R)
##@btime op3($A, $R)
##@show op1(A, R) 
##@show op3(A, R)
##
##X = similar(R) 
##
##using LinearAlgebra, SparseArrays, BenchmarkTools
##function mymul!(
##    Yt::{Tv},
##    A::SparseMatrixCSC{Tv},
##    X::AbstractMatrix{Tv},
##    α::Tv,
##    β::Tv,
##    ) where {Tv <: AbstractFloat}
##    lmul!(β, Y)
##    Yt = Y'
##    Xt = X'
##    if (size(Y, 1) != size(A, 1) || size(X, 1) != size(A, 2) || size(Y, 2) != size(X, 2))
##        throw(DimensionMismatch("dimension mismatch"))
##    end
##    @inbounds for (x, y, v) in zip(findnz(A)...)
##        @view(Yt[:, x]) .+= α * v * @view(Xt[:, y]) 
##    end
##end
##
##Y = X + A*R
##Z = deepcopy(X)
##
##mymul!(X, A, R, 1.0, 1.0)
##mul!(Z, A, R, 1.0, 1.0)
##@show norm(Y - X)
##@show norm(Z - X)