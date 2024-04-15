using SparseArrays
using LinearAlgebra
using BenchmarkTools

BLAS.set_num_threads(1)

using Plots
n = 10^6
m = 10^8
A = sprand(n, n, m / n^2)
v = randn(n)

At = sparse(A')

@btime $A * $v
@btime $v' * $At
@btime $v * $At 

@edit A * v

function myAv(A, v)
    n = size(A, 1)
    u = zeros(eltype(A), n) 
    nzval = A.nzval
    colptr = A.colptr
    rowval  = A.rowval
    for col = 1:n
      @inbounds for nzi = colptr[col]:colptr[col+1]-1
        row = rowval[nzi]
        val = nzval[nzi]
        u[row] += val * v[col] 
      end
    end
    return u
end

function myvAt(v, A)
    n = size(A, 1)
    u = zeros(eltype(A), n) 
    nzval = A.nzval
    colptr = A.colptr
    rowval  = A.rowval
    for col = 1:n
      @inbounds for nzi = colptr[col]:colptr[col+1]-1
        row = rowval[nzi]
        u[col] += nzval[nzi] * v[row] 
      end
    end
    return u
end

@btime myAv($A, $v) 
@btime myvAt($v, $At)
@btime C = SparseMatrixCSC(($A).m, ($A).n, ($A).colptr, ($A).rowval, ($A).nzval)
