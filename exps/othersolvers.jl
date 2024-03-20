function maxcut_sdp(
    A::SparseMatrixCSC{Tv, Ti},
    eps::Tv,
    solver::String="Mosek",
) where {Tv <: AbstractFloat, Ti <: Integer}
    n = size(A, 1)
    D = Diagonal(vec(sum(A, dims=1)))
    L = sparse(D) - A
    X = Semidefinite(n)

    problem = maximize(sum(L .* X), diag(X) == 1, isposdef(X))
    if solver == "SCS"
        opt = MOI.OptimizerWithAttributes(SCS.Optimizer,
                                         "max_iters" => 1000000, "verbose" => 1,
                                         "eps_abs" => eps, "eps_rel" => eps)
    elseif solver == "Mosek"
        opt = MOI.OptimizerWithAttributes(Mosek.Optimizer,
                                         "QUIET" => false, 
                                         "INTPNT_CO_TOL_DFEAS" => eps, 
                                         "INTPNT_CO_TOL_PFEAS" => eps,
                                         )
    end
    solve!(problem, opt)
    return problem.optval
end