@with_kw mutable struct BurerMonteiroConfig{Ti<:Integer,Tv}
    ptol::Tv = 1e-2                    # primal infeasibility tolerance 
    gtol::Tv = 0.0                     # stationarity tolerance
    objtol::Tv = 1e-2                  # suboptimality tolerance
    σ_0::Tv = 2.0                      # initial penalty parameter
    σfac::Tv = 2.0                     # factor for increasing σ
    maxtime::Tv = 3600.0               # maximum time in seconds(better use terminal tool for limiting time)
    printlevel::Ti = 1                 # print level
    printfreq::Tv = 60.0               # how often to print in seconds
    numlbfgsvecs::Ti = 4               # number of L-BFGS vectors
    maxmajoriter::Ti = 10^5            # maximum number of major iterations
    maxiter::Ti = 10^7                 # maximum number of total iterations
    fprec::Tv = 1e8                    # for moderate accuracy
    rankupd_tol::Ti = 4                # rank update tolerance
    prior_trace_bound::Tv = 1e18       # a trace bound determined or estimated
    dataset::String = ""               # dataset name
    eval_DIMACS_errs::Bool = false     # whether to evaluate DIMACS errors
    eigval_highprecision = false       # GenericArpack (true) / Lanczos with random start (false)
    init_func::Union{Nothing,Function} = nothing  # custom init; used in rank_update! too
    init_args::Tuple = ()              # extra args for init_func
    gtol_mode::Symbol = :relative      # gradient norm mode: :relative (÷ normC) or :absolute
    ptol_mode::Symbol = :relative      # primal vio mode: :relative (÷ normb) or :absolute
    objtol_mode::Symbol = :relative    # obj gap mode: :relative (÷ min|obj,dual|) or :absolute
end
