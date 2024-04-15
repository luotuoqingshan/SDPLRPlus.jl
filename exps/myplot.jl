using BenchmarkProfiles 

T = 10 * rand(3, 3) .+ 10;

@show T

using Plots
performance_profile(
    PlotsBackend(), 
    T, 
    ["Solver 1", "Solver 2", "Solver 3"],
    title="Max Cut",
)