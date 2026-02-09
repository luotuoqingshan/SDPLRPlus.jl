using Revise
using SparseArrays
using SDPLRPlus

using LowRankOpt
using Dualization
include(joinpath(dirname(dirname(pathof(LowRankOpt))), "examples", "maxcut.jl"))

n = 500
import Random
Random.seed!(0)
W = sprand(n, n, 0.01)
W = W + W'
include(joinpath(@__DIR__, "problems.jl"))
C, As, b = maxcut(W)
@time sdplr(C, As, b, 1, maxmajoriter = 20);

# SDPLRPlus does not support sparse factor
As = [SymLowRankMatrix(Diagonal(ones(1)), hcat(e_i(Float64, i, n, sparse = false, vector = false))) for i in 1:n]
d = SDPLRPlus.SDPData(C, As, b)
var = SDPLRPlus.SolverVars(d, 1)
aux = SDPLRPlus.SolverAuxiliary(d)
@time sdplr(C, As, b, 1, maxmajoriter = 50);

model = maxcut(W, dual_optimizer(LRO.Optimizer))
set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
set_attribute(model, "sub_solver", SDPLRPlus.Solver)
set_attribute(model, "ranks", [1])
set_attribute(model, "maxmajoriter", 0)
set_attribute(model, "printlevel", 3)
@profview optimize!(model)
solver = unsafe_backend(model).dual_problem.dual_model.model.optimizer.solver

using BenchmarkTools
function A_sym_bench(aux, var, nlp, var_lro)
    println("𝒜 sym")
    @btime SDPLRPlus.𝒜!(var.primal_vio, aux, var.Rt)
    @btime SDPLRPlus.𝒜!(var_lro.primal_vio, nlp, var_lro.Rt)
end
function A_not_sym_bench(aux, var, nlp, var_lro)
    println("𝒜 not sym")
    @btime SDPLRPlus.𝒜!(var.A_RD, aux, var.Rt, var.Gt)
    @btime SDPLRPlus.𝒜!(var_lro.A_RD, nlp, var_lro.Rt, var_lro.Gt)
end
function At_bench(aux, var, nlp, var_lro)
    println("𝒜t")
    @btime SDPLRPlus.𝒜t!($var.Gt, $var.Rt, $aux, $var)
    @btime SDPLRPlus.𝒜t!($var_lro.Gt, $var_lro.Rt, $nlp, $var_lro)
end
function At_bench2(aux, var, nlp, var_lro)
    println("𝒜t rank-1")
    x = rand(n)
    y = similar(x)
    @time SDPLRPlus.𝒜t!(y, aux, x, var)
    @time SDPLRPlus.𝒜t!(y, nlp, x, var_lro)
end
nlp = solver.model
var_lro = solver.solver.var
A_sym_bench(aux, var, solver.model, solver.solver.var)
A_not_sym_bench(aux, var, solver.model, solver.solver.var)
At_bench(aux, var, solver.model, solver.solver.var)
At_bench2(aux, var, solver.model, solver.solver.var)
@profview SDPLRPlus.𝒜!(var.primal_vio, aux, var.Rt)
@profview for i in 1:100
    SDPLRPlus.𝒜!(var_lro.primal_vio, nlp, var_lro.Rt)
end
@time SDPLRPlus.𝒜!(var_lro.primal_vio, nlp, var_lro.Rt)
@time SDPLRPlus.𝒜t!(var_lro.Gt, var_lro.Rt, nlp, var_lro)
@profview for _ in 1:1000
    SDPLRPlus.𝒜t!(var_lro.Gt, var_lro.Rt, nlp, var_lro)
end
function bench_lmul(A)
    n = LinearAlgebra.checksquare(A)
    x = rand(1, n)
    y = similar(x)
    @profview for i in 1:100000
        LinearAlgebra.mul!(y, x, A, 2.0, 1.0)
    end
    #@btime LinearAlgebra.mul!($y, $x, $A, 2.0, 1.0)
end
function bench_rmul(A)
    n = LinearAlgebra.checksquare(A)
    x = rand(n, 1)
    y = similar(x)
    @btime LinearAlgebra.mul!($y, $A, $x, 2.0, 1.0)
end
bench_lmul(aux.symlowrank_As[1]);
A = aux.symlowrank_As[1]
n = LinearAlgebra.checksquare(A)
x = rand(1, n)
y = similar(x)
LinearAlgebra.mul!(y, x, A, 2.0, 1.0)

bench_rmul(nlp.model.A[1]);
bench_lmul(nlp.model.A[1])
A = nlp.model.A[1]
n = LinearAlgebra.checksquare(A)
x = rand(n, 1)
y = similar(x)
@time LinearAlgebra.mul!(y, A, x, 2.0, 1.0);
@btime LinearAlgebra.mul!($y, $A, $x, 2.0, 1.0);
methods(LinearAlgebra.mul!, typeof.((y, A, x, 2.0, 1.0)))

A = nlp.model.A[1]
n = LinearAlgebra.checksquare(A)
x = rand(n)
y = similar(x)
x = rand(n)
y = similar(x)
@btime LinearAlgebra.mul!($y, $A, $x, 2.0, 1.0);

C = LRO._lmul_diag!!(A.scaling, LRO.right_factor(A)' * x)
lA = LRO.left_factor(A)
@btime LinearAlgebra.mul!($y, $C, $lA)
@edit LinearAlgebra.mul!(y, lA, C);
@edit LinearAlgebra.mul!(y, lA, C, true, true);
@edit LinearAlgebra._rscale_add!(y, lA, C, true, true);
@edit LinearAlgebra.mul!($y, $lA, $C, 2.0, 1.0);
