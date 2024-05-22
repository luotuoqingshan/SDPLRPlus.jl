@with_kw mutable struct BurerMonteiroConfig{Ti <: Integer, Tv}
    ptol::Tv = 1e-2
    objtol::Tv = 1e-2
    σfac::Tv = 2.0 
    maxtime::Tv = 3600.0
    printlevel::Ti = 1
    printfreq::Tv = 60.0
    numlbfgsvecs::Ti = 4 
    maxmajoriter::Ti = 10^5
    maxiter::Ti = 10^7
    fprec::Tv = 1e8 # for moderate accuracy
    rankupd_tol::Ti = 4
    prior_trace_bound::Tv = 1e18
    dataset::String = ""
    eval_DIMACS_errs::Bool = false
end