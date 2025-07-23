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

    r = src.dim.ranks[]
    m = src.meta.ncon
    Tv = Float64

    var = SolverVars(src, r)

    stats = SolverStats(
        Ref(zero(Tv)), # starttime
        Ref(zero(Tv)), # endtime
        Ref(zero(Tv)), # time spent on lanczos with random start
        Ref(zero(Tv)), # time spent on GenericArpack
        Ref(zero(Tv)), # primal time
        Ref(zero(Tv)), # DIMACS time
    )

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
    λ0 = randn(m)
    return SolverVars(Rt0, λ0, r)
end

function 𝒜!(
    𝒜_UUt::Vector{Tv},
    model::LRO.BurerMonteiro.Model,
    Ut::Matrix{Tv},
) where {Tv}
    fill!(𝒜_UUt, zero(Tv))
    m = model.meta.ncon
    v = vec(Ut')
    NLPModels.cons!(model, v, view(𝒜_UUt, 1:m))
    𝒜_UUt[end] = NLPModels.obj(model, v)
end

function 𝒜t_preprocess!(::SolverVars, ::LRO.BurerMonteiro.Model) end

function 𝒜t!(
    y::Matrix,
    x::Matrix,
    model::LRO.BurerMonteiro.Model,
    var::SolverVars,
)
    fill!(y, zero(Tv))
    v = vec(x')
    NLPModels.jtprod!(model, x::AbstractVector, y::AbstractVector, Jtv::AbstractVector)
end

function 𝒜t!(
    y::Tx, 
    x::Tx, 
    aux::SolverAuxiliary{Ti, Tv}, 
    var::SolverVars{Ti, Tv},
)where{Ti <: Integer, Tv, Tx <: AbstractArray{Tv}}


    # then deal with low-rank matrices 
    if aux.n_symlowrank_matrices > 0
        for i = 1:aux.n_symlowrank_matrices
            global_id = aux.symlowrank_As_global_inds[i]
            coeff = var.y[global_id]
            mul!(y, x, aux.symlowrank_As[i], coeff, one(Tv))
        end 
    end
end
