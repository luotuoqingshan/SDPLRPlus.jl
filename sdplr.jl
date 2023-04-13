include("structures.jl")
include("dataoper.jl")
include("lbfgs.jl")

"""
Interface for SDPLR 

Problem formulation:
    min   Tr(C (YY^T))
    s.t.  Tr(As[i] (YY^T)) = bs[i]
            Y in R^{n x r}
"""
function sdplr(
    C::Matrix{Float64},
    As::Vector{AbstractMatrix},
    bs::Vector,
    r::Int;
)
    m = length(As)
    @assert (typeof(C) <: SparseMatrixCSC 
         || typeof(C) <: Diagonal
         || typeof(C) <: LowRankMatrix) "Wrong matrix type of cost matrix."
    pdata = ProblemData(m, 0, 0, 0,
                        SparseMatrixCSC[],
                        Diagonal[],
                        LowRankMatrix[],
                        C, bs)
    for i = 1:m
        if typeof(As[i]) <: SparseMatrixCSC
            push!(pdata.A_sp, As[i])
            pdata.m_sp += 1
        elseif typeof(As[i]) <: Diagonal
            push!(pdata.A_d, As[i])
            pdata.m_d += 1 
        elseif typeof(As[i]) <: LowRankMatrix
            push!(pdata.A_lr, As[i])
            pdata.m_lr += 1
        else
            error("Wrong matrix type of constraint matrix.")
        end
    end
end


function _sdplr(
    pdata::ProblemData,
    algdata::AlgorithmData,
    R::Matrix{Float64},
    config::Config,
)
    # TODO setup config
    # TODO create all data structures

    # TODO double check scaling 

    # misc declarations
    recalcfreq = 5 
    difficulty = 3 
    bestinfeas = 1.0e10

    # TODO setup printing
    if config.printlevel > 0
        printheading()
    end


    # set up algorithm parameters
    normb = norm(pdata.b, Inf)
    normC = C_normdatamat(pdata)
    best_dualbd = -1.0e20

    # initialize lbfgs datastructures
    lbfgshis = lbfgshistory(
        config.numberfgsvecs,
        lbfgsvec[],
        0)

    for i = 1:config.numberfgsvecs
        push!(lbfgshis.vecs, lbfgsvec(zeros(size(R)), zeros(size(R)), 0.0, 0.0))
    end


    # TODO essential_calc
    rho_c_tol = config.rho_c / algdata.sigma 
    val, rho_c_val, rho_f_val = essential_calcs!(pdata, algdata, R, normC, normb)
    majiter = config.majiter

    # save initial function value, notice that
    # here the constraints may not be satisfied,
    # which means the value may be smaller than the optimum
    origval = val 

    majoriter_end = true

    while majiter < 10^5 
        #avoid goto in C
        current_majoriter_end = false

        while((!config.sigmastrategy && lambdaupdate < lambdaupdatect)
            ||(config.sigmastrategy && difficulty != 1)) 

            # increase lambda counter, reset local iter counter and lastval
            lambdaupdate += 1
            localiter = 0
            lastval = 1.0e10

            # check stopping criteria: rho_c_val = norm of gradient
            if rho_c_val <= rho_c_tol
                break
            end

            # in the local iteration, we keep optimizing
            # the subproblem using lbfgsb and return a solution
            # satisfying stationarity condition
            while(rho_c_val > rho_c_tol) 
                #increase both iter and localiter counters
                iter += 1
                localiter += 1
                # direction has been negated
                dir = dirlbfgs(algdata, lbfgshis, negate=1)

                if (dir .* algdata.G) >= 0 # not a descent direction
                    dir = -algdata.G # reverse back to gradient direction
                end

                lastval = val
                alpha, val = linesearch!(pdata, algdata, R, D, 
                                         dir, alpha_max=1.0, update=1) 

                if recalc_cnt == 0
                    val, rho_c_val, rho_f_val = 
                        essential_calcs!(pdata, algdata, R, normC, normb)
                    recalc_cnt = recalcfreq
                else
                    gradient!(pdata, algdata, R)
                    rho_c_val = norm(algdata.G, "fro") / (1.0 + normC)
                    rho_f_val = norm(algdata.vio, 2) / (1.0 + normb)
                    recalc_cnt -= 1
                end

                if algdata.numbfgsvecs > 0 
                    lbfgs_postprocess!(algdata, lbfgshis, dir, alpha)
                end

                if (algdata.totaltime >= config.timelimit 
                    || rho_f_val <= config.rho_f
                    ||  iter >= 10^7)
                    current_majoriter_end = true
                    break
                end

                bestinfeas = min(rho_f_val, bestinfeas)
            end

            if current_majoriter_end
                majoriter_end = true
                break
            end

            # update Lagrange multipliers and recalculate essentials
            algdata.lambda = -algdata.sigma * algdata.vio
            val, rho_c_val, rho_f_val = 
                essential_cals!(pdata, algdata, R, normC, normb)

            if SIGMASTRATEGY
                if localiter <= 10
                    difficulty = 1 # EASY
                elseif localiter > 10 && localiter <= 50 
                    difficulty = 2 # MEDIUM
                else
                    difficulty = 3 # HARD
            end
            # TODO check dual bounds
        end # end one major iteration

        # cannot further improve infeasibility,
        # in other words to make the solution feasible, 
        # we get a ridiculously large value
        if val > 1.0e10 * fabs(origval) 
            majoriter_end = true
            printf("Cannot reduce infeasibility any further.\n")
        end

        if isnan(val)
            println("Error(sdplrlib): Got NaN.")
            return 0
        end

        # TODO potential rank reduction 

        # update sigma
        while true
            algdata.sigma *= config.sigmafac
            val, rho_c_val, rho_f_val = 
                essential_calcs!(pdata, algdata, R, normC, normb)
            rho_c_tol = config.rho_c / algdata.sigma
            if rho_c_tol < rho_c_val
                break
            end
        end
        # refresh some parameters
        lambdaupdate = 0
        if config.sigmastrategy
            difficulty = 3
        end

        # clear bfgs vectors
        for i = 1:lbfgshis.m
            lbfgshis.vecs[i] = lbfgsvec(zeros(size(R)), zeros(size(R)), 0.0, 0.0)
        end
    end
end