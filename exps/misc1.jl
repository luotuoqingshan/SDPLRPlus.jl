using MAT, JSON

res = matread(homedir()*"/maxcut/rudytest.mat")
res = res["record"]

ind = [(1:67)..., 70, 72, 77, 81]

manopt_dt = res[ind, 4, 1]
manopt_rel_duality_bd = res[ind, 5, 1]
manopt_incr_dt = res[ind, 4, 2]
manopt_incr_rel_duality_bd = res[ind, 5, 2]

sdplr_dt = Float64[]
sdplr_rel_duality_bd = Float64[]
for i = ind 
    file = homedir()*"/SDPLR-jl/output/MaxCut/G$i/SDPLR-jl-seed-0.json"
    res = JSON.parsefile(file)
    push!(sdplr_dt, res["primaltime"])
    push!(sdplr_rel_duality_bd, res["rel_duality_bound"])
end

using DataFrames
graphs = ["G$i" for i = ind]
df = DataFrame(Graphs=graphs, Manopt_time=manopt_dt, Manopt_rel_duality_gap=manopt_rel_duality_bd,
               Manopt_incr_time=manopt_incr_dt, Manopt_incr_rel_duality_gap=manopt_incr_rel_duality_bd,
               SDPLR_time=sdplr_dt, SDPLR_rel_duality_gap=sdplr_rel_duality_bd,)

using CSV
using Plots
T = reduce(hcat, (manopt_dt, manopt_incr_dt, sdplr_dt))
solvers = ["Manopt", "Manopt_incr", "SDPLR"]
using BenchmarkProfiles 
performance_profile(PlotsBackend(), T, solvers, title="Manopt vs SDPLR on MaxCut")
savefig(homedir()*"/maxcut/manopt_vs_sdplr.pdf")
CSV.write(homedir()*"/maxcut/manopt_vs_sdplr.csv", df)



## real-world graphs
res = matread(homedir()*"/maxcut/rudytest_realworldgraphs.mat")
res = res["record"]

graphs = ["ca-AstroPh", "ca-CondMat", "ca-GrQc", "ca-HepPh", "ca-HepTh", "email-Enron", "deezer_europe", "musae_facebook"]
ind = 1:length(graphs) 

manopt_dt = res[ind, 4, 1]
manopt_rel_duality_bd = res[ind, 5, 1]
manopt_incr_dt = res[ind, 4, 2]
manopt_incr_rel_duality_bd = res[ind, 5, 2]

sdplr_dt = Float64[]
sdplr_rel_duality_bd = Float64[]
for graphname = graphs 
    file = homedir()*"/SDPLR-jl/output/MaxCut/$graphname/SDPLR-jl-seed-0.json"
    res = JSON.parsefile(file)
    push!(sdplr_dt, res["primaltime"])
    push!(sdplr_rel_duality_bd, res["rel_duality_bound"])
end

df = DataFrame(Graphs=graphs, Manopt_time=manopt_dt, Manopt_rel_duality_gap=manopt_rel_duality_bd,
               Manopt_incr_time=manopt_incr_dt, Manopt_incr_rel_duality_gap=manopt_incr_rel_duality_bd,
               SDPLR_time=sdplr_dt, SDPLR_rel_duality_gap=sdplr_rel_duality_bd,)

T = reduce(hcat, (manopt_dt, manopt_incr_dt, sdplr_dt))
solvers = ["Manopt", "Manopt_incr", "SDPLR"]
using BenchmarkProfiles 
performance_profile(PlotsBackend(), T, solvers, title="Manopt vs SDPLR on MaxCut")
savefig(homedir()*"/maxcut/manopt_vs_sdplr_moderate_realworld_graphs.pdf")
CSV.write(homedir()*"/maxcut/manopt_vs_sdplr_moderate_realworld_graphs.csv", df)