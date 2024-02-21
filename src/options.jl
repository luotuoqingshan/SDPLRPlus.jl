@with_kw struct BurerMonteiroConfig{Ti <: Integer, Tv <: AbstractFloat}
    stationarity_tol::Tv = 1e-3
    primal_vio_tol::Tv = 1e-3
    duality_gap_tol::Tv = 1e-3
    sigma_fac::Tv = 2.0 
    #rankreduce::Int
    time_limit::Tv = 3600.0
    printlevel::Ti = 1
    printfreq::Tv = 60.0
    numlbfgsvecs::Ti = 4 
    majoriter_limit::Ti = 10^5
    iter_limit::Ti = 10^7
    checkdual::Bool = true 
end