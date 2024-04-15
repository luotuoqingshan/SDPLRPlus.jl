using BenchmarkTools, SparseArrays, LinearAlgebra

BLAS.set_num_threads(1)

a = randn()
n = 10^5
b = randn(n+1)
b1 = b[1:n]
σ = 2.0
c = randn(n)

@btime axpy!($σ, $b1, $c)
v = @view b[1:n]
@btime axpy!($σ, $v, $c)