using Revise
using SparseArrays
using SDPLRPlus
n = 5
import Random
Random.seed!(0)
W = sprand(n, n, 0.1)
W = W + W'
include(joinpath(@__DIR__, "problems.jl"))
C, As, b = maxcut(W)
@time sdplr(C, As, b, 1, maxmajoriter = 20);
function e_i(i, n)
    ei = zeros(n, 1)
    ei[i] = 1
    return ei
end
As = [SymLowRankMatrix(Diagonal(ones(1)), e_i(i, n)) for i in 1:n]
@time sdplr(C, As, b, 1, maxmajoriter = 50);

using LowRankOpt
using Dualization
include(joinpath(dirname(dirname(pathof(LowRankOpt))), "examples", "maxcut.jl"))
model = maxcut(W, dual_optimizer(LRO.Optimizer))
set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
set_attribute(model, "sub_solver", SDPLRPlus.Solver)
set_attribute(model, "ranks", [1])
set_attribute(model, "maxmajoriter", 10)
set_attribute(model, "printlevel", 3)
optimize!(model)
