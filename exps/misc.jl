using SparseArrays
using MKL
using MKLSparse


using BenchmarkTools
#using MAT
#using LinearAlgebra
#using MKLSparse
#
#
MKL.BLAS.set_num_threads(1)
#
n = 10^4 
m = 10^7
r = 20
A = sprand(n, n, m / n^2) 
A = max.(A, A')
R = randn(n, r)
Rt = Matrix(R')
@btime $A * $R;
@btime $Rt * $A;

MKL.axpy!(1.0, R, R)
#matwrite(homedir()*"/matmul_data.mat",
#    Dict("A" => A, "R" => R, "Rt" => Rt))
#
#@btime $A * $B;
#
#@btime $A * $R;
#@btime $Rt * $A;


#@btime $A * $R;
#@btime mul!($output, $A, $R);
#@btime $Rt * $A; 
#
#
#output = zeros(1,r)
#@btime $Bt * $R
#@btime mul!($output, $Bt, $R)
#
#ratio = m / n^2
#B = sprand(n, n, ratio)
#SymB = max.(B, B')
#
#R = randn(n, r)
#Rt = Matrix(R')
#
#C = zeros(n, r)
#Ct = zeros(r, n)
#R1 = randn(n, r) 
#
#@btime dot($R, $R1)
#
#BLAS.set_num_threads(4)
#@btime dot($R, $R1)
#
#
#@btime mul!($C, $SymB, $R)
#@btime mul!($Ct, $Rt, $SymB)
#@which mul!(C, SymB, R)
#@which mul!(C, Rt, SymB)
#
#using MKLSparse
#@btime mul!($C, $SymB, $R)
#@btime mul!($Ct, $Rt, $SymB)
#@which mul!(C, SymB, R)
#@which mul!(Ct, Rt, SymB)

#function myaxpy!(a, x, y)
#  @simd for i = axes(x, 1) 
#    @inbounds y[i] += a * x[i]
#  end
#end
#
#
using SparseArrays
using BenchmarkTools
using LinearAlgebra
n = 10^3
m = 10^5
r = 20
A = sprand(n, n, m / n^2)
A = max.(A, A')
R = randn(n, r)
Rt = Matrix(R')
Ct = zeros(r, n)
C = zeros(n, r)
function myprod!(Ct, Rt, SymA)
  r, n = size(Rt)
  fill!(Ct, zero(eltype(Ct)))
  nzval = SymA.nzval
  colptr = SymA.colptr
  rowval  = SymA.rowval
  for col = 1:n
    for nzi = colptr[col]:colptr[col+1]-1
      row = rowval[nzi]
      val = nzval[nzi]
      @simd for i = 1:r
        @inbounds Ct[i, col] += val * Rt[i, row]
      end
    end
  end
end
#using MKLSparse
@btime mul!($C, $A, $R);
@btime mul!($Ct, $Rt, $A);
@btime myprod!($Ct, $Rt, $A);
#
#funtion myprod!(C, A, B)
#  m, n = size(A)
#  k = size(B, 2)
#  for i = 1:n
#    @simd @inbounds for nzi = A.colptr[i]:A.colptr[i+1]-1
#      row = A.rowval[nzi]
#    end
#end

#function mydot(Rt, row, col)
#    m = size(Rt, 1)
#    rval = zero(eltype(Rt))
#    @simd for i in 1:m
#      @inbounds rval += Rt[i, row] * Rt[i, col]
#    end 
#    return rval 
#end 
#
#function mydot_view(Rt, row, col)
#    return dot(@view(Rt[:, row]), @view(Rt[:, col]))
#end
#  
#function f(A, is, Rt) 
#  res = zero(eltype(Rt))
#  for i in is 
#    for nzi in A.colptr[i]:A.colptr[i+1]-1
#      row = A.rowval[nzi]
#      res += A.nzval[nzi] * mydot(Rt, row, i)
#    end
#  end 
#  return res
#end 
#  
#function f_view(A, is, Rt) 
#  res = zero(eltype(Rt))
#  for i in is 
#    for nzi in A.colptr[i]:A.colptr[i+1]-1
#      row = A.rowval[nzi]
#      res += A.nzval[nzi] * mydot_view(Rt, row, i)
#    end
#  end 
#  return res
#end
#  
#  
#function f(A, Rt)
#  chunks = Iterators.partition(axes(A,2), length(axes(A,2)) ÷ Threads.nthreads())
#  tasks = map(chunks) do chunk
#      Threads.@spawn f(A, chunk, Rt)
#  end
#  return sum(fetch, tasks)
#end
#
#function f_view(A, Rt)
#  chunks = Iterators.partition(axes(A,2), length(axes(A,2)) ÷ Threads.nthreads())
#  tasks = map(chunks) do chunk
#      Threads.@spawn f_view(A, chunk, Rt)
#  end
#  return sum(fetch, tasks)
#end
#
#@btime dot($R, $SymB * $R) 
#@btime f($SymB, $Rt) 
#@btime f_view($SymB, $Rt)
