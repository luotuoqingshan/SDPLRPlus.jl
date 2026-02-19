# Run tests from package root with: julia --project -e 'using Pkg; Pkg.test()'
# That uses the local SDPLRPlus source (no need to add the package from the registry).
using SDPLRPlus:
    sdplr,
    linesearch!,
    g!,
    SDPData,
    SymLowRankMatrix,
    SolverVars,
    SolverAuxiliary,
    BurerMonteiroConfig,
    f!,
    𝒜t!,
    𝒜t_preprocess!
using Test

using Random
using LuxurySparse, SparseArrays, LinearAlgebra
using FiniteDiff

function make_random_graph(n, p)
    A_dense = rand(n, n)
    A_dense = (A_dense .+ A_dense') / 2
    A_dense = Float64.(A_dense .> p)
    A_dense[diagind(A_dense)] .= 0.0
    return sparse(A_dense)
end

# write tests here
include("symlowrank.jl")

include("problem.jl")

include("maxcut.jl")

include("minimumbisection.jl")

include("lovasztheta.jl")

include("cutnorm.jl")

include("coreop.jl")

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
