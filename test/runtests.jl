using SDPLRPlus: sdplr, linesearch!, g!, SDPData, SymLowRankMatrix, SolverVars, SolverAuxiliary, BurerMonteiroConfig, f!
using Test

using Random
using LuxurySparse, SparseArrays, LinearAlgebra

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
