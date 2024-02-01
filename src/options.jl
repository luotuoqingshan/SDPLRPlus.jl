@with_kw struct BurerMonteiroConfig{Ti <: Integer, Tv <: AbstractFloat}
    tol_stationarity::Tv = 1e-1
    tol_primal_vio::Tv = 1e-4
    σ_fac::Tv = 2.0 
    #rankreduce::Int
    timelim::Tv = 3600.0
    printlevel::Ti = 1
    printfreq::Tv = 60.0
    numlbfgsvecs::Ti = 4 
    σ_strategy::Ti = 1
    λ_updatect::Ti = 1
    #rankredtol::Float64
    #gaptol::Float64
    maxmajoriter::Ti = 10^5
    maxiter::Ti = 10^7
    checkdual::Bool = true 
end