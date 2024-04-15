using BenchmarkTools
using SparseArrays
using LinearAlgebra 

n = 10^5
r = 100
s = 10

Bt = randn(s, n)
R = randn(n, r)
C = randn(s, r)
@btime mul!($C, $Bt, $R)
@btime $Bt * $R
