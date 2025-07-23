import SolverCore, NLPModels

struct Solver <: SolverCore.AbstractOptimizationSolver
    var::SolverVars{Int,Float64}
    stats::SolverStats{Float64}
    config::BurerMonteiroConfig{Int,Float64}
end

function Solver(src::NLPModels.AbstractNLPModel; config=BurerMonteiroConfig{Int,Float64}(), kwargs...)
    for (key, value) in kwargs
        if hasfield(BurerMonteiroConfig, Symbol(key))
            setfield!(config, Symbol(key), value)
        else
            @error "Unrecognized keyword argument $key"
        end
    end 

    var = SolverVars(src, src.dim.ranks[])
    stats = SolverStats{Float64}()
    return Solver(var, stats, config)
end

function SolverCore.solve!(
    solver::Solver,
    model::NLPModels.AbstractNLPModel, # Same the one given in the constructor
    stats::SolverCore.GenericExecutionStats;
    kws...,
)
    return _sdplr(model, solver.var, model, solver.stats, solver.config)
end

import LowRankOpt as LRO

b_vector(model::LRO.BurerMonteiro.Model) = LRO.cons_constant(model.model)
C_matrix(model::LRO.BurerMonteiro.Model) = NLPModels.grad(model.model, LRO.MatrixIndex(1))

barvinok_pataki(data::LRO.BurerMonteiro.Model) = barvinok_pataki(data.dim.side_dimension[], data.meta.ncon)

function SolverVars(data::LRO.BurerMonteiro.Model, r)
    # randomly initialize primal and dual variables
    Rt0 = 2 .* rand(length(data.dim)) .- 1
    λ0 = randn(data.meta.ncon)
    return SolverVars(Rt0, λ0, r)
end

function 𝒜!(
    𝒜_UUt::Vector{Tv},
    model::LRO.BurerMonteiro.Model,
    x::Vector{Tv},
) where {Tv}
    m = model.meta.ncon
    NLPModels.cons!(model, x, view(𝒜_UUt, 1:m))
    𝒜_UUt[end] = NLPModels.obj(model, x)
end

function 𝒜!(
    𝒜_UVt::Vector{Tv},
    model::LRO.BurerMonteiro.Model,
    u::Vector{Tv},
    v::Vector{Tv},
) where {Tv}
    m = model.meta.ncon
    NLPModels.jprod!(model, u, v, view(𝒜_UVt, 1:m))
    𝒜_UVt[end] = LRO.BurerMonteiro.gprod(model, u, v)
    return
end

function 𝒜t_preprocess!(::SolverVars, ::LRO.BurerMonteiro.Model) end

function 𝒜t!(
    Jtv::Vector,
    x::Vector,
    model::LRO.BurerMonteiro.Model,
    var::SolverVars,
)
    y = view(var.y, 1:model.meta.ncon)
    NLPModels.jtprod!(model, x, y, Jtv)
    Jtv ./= 2
    Jtv .+= NLPModels.grad(model, x) / 2
    return
end
