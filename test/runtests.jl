# Run tests from package root with: julia --project -e 'using Pkg; Pkg.test()'
# That uses the local SDPLRPlus source (no need to add the package from the registry).
using SDPLRPlus: sdplr, linesearch!, g!, SDPData, SymLowRankMatrix, SolverVars, SolverAuxiliary, BurerMonteiroConfig, f!
using Test

using Random
using LuxurySparse, SparseArrays, LinearAlgebra
using FiniteDiff

function primal_vio(C, As, bs, Rt)
  m = length(bs)
  primal_vio = zeros(Float64, m + 1)
  primal_vio[m+1] = tr(Rt * C * Rt') # objective
  for i in 1:m
    primal_vio[i] = tr(Rt * As[i] * Rt') - bs[i]
  end
  return primal_vio
end

function make_random_graph(n, p)
    A_dense = rand(n, n)
    A_dense = (A_dense .+ A_dense') / 2
    A_dense = Float64.(A_dense .> p)
    A_dense[diagind(A_dense)] .= 0.0
    return sparse(A_dense)
end

function test_gradient_fd!(data, var, aux)
    r, n = size(var.Rt)
    rt_vec = vec(copy(var.Rt))
    ℒ_scalar(x::Vector) = (copyto!(var.Rt, reshape(x, r, n)); f!(data, var, aux))
    grad_num = FiniteDiff.finite_difference_gradient(ℒ_scalar, copy(rt_vec))
    copyto!(var.Rt, reshape(rt_vec, r, n))
    f!(data, var, aux)
    g!(var, aux)
    grad_ana = vec(var.Gt)
    rel_err = norm(grad_num - grad_ana, Inf) / (1 + norm(grad_ana, Inf))
    @test rel_err < 1e-8
end

# write tests here
include("symlowrank.jl")

include("problem.jl")

include("maxcut.jl")

include("minimumbisection.jl")

include("lovasztheta.jl")

include("cutnorm.jl")

## NOTE add JET to the test environment, then uncomment
# using JET
# @testset "static analysis with JET.jl" begin
#     @test isempty(JET.get_reports(report_package(SDPLR, target_modules=(SDPLR,))))
# end

## NOTE add Aqua to the test environment, then uncomment
# @testset "QA with Aqua" begin
#     import Aqua
#     Aqua.test_all(SDPLR; ambiguities = false)
#     # testing separately, cf https://github.com/JuliaTesting/Aqua.jl/issues/77
#     Aqua.test_ambiguities(SDPLR)
# end
