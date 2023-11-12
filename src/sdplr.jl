using GenericArpack
using MKLSparse # to speed up sparse matrix multiplication
using Random
using SparseArrays
include("structs.jl")
include("dataoper.jl")
include("lbfgs.jl")
include("myprint.jl")
include("linesearch.jl")
include("options.jl")


"""
    sparse_sym_to_upper_tri(I, J, V)

Convert a symmetric matrix with form (I, J, V) to upper triangular form.
"""
function sparse_sym_to_upper_tri(
    I::Vector{Ti}, 
    J::Vector{Ti},
    V::Vector{Tv},
)where{Ti <: Integer, Tv <: AbstractFloat}
    upper_tri_I, upper_tri_J, upper_tri_V = Ti[], Ti[], Tv[]
    for i in eachindex(I)
        # only store upper triangular part
        if I[i] <= J[i] 
            push!(upper_tri_I, I[i])
            push!(upper_tri_J, J[i])
            push!(upper_tri_V, V[i])
        end
    end
    return upper_tri_I, upper_tri_J, upper_tri_V
end


"""
    preprocess_sparsecons(As)

Preprocess the sparse constraints to initialize necessary 
data structures for BurerMonteiro algorithm.
"""
function preprocess_sparsecons(
    As::Vector{SparseMatrixCSC{Tv, Ti}}
) where {Tv <: AbstractFloat, Ti <: Integer}
    # aggregate all constraints into one sparse matrix
    all_I, all_J, all_V = Ti[], Ti[], Tv[]
    n = size(As[1], 1)
    nA = length(As)
    total_nnz = 0
    upper_tri_I_list = Vector{Ti}[]
    upper_tri_J_list = Vector{Ti}[]
    upper_tri_V_list = Vector{Tv}[]
    for A in As
        AI, AJ, AV = findnz(A)
        upper_tri_AI, upper_tri_AJ, upper_tri_AV = 
            sparse_sym_to_upper_tri(AI, AJ, AV)
        push!(upper_tri_I_list)
        push!(upper_tri_J_list)
        push!(upper_tri_V_list)
        append!(all_I, upper_tri_AI)
        append!(all_J, upper_tri_AJ)
        # make sure cancellation doesn't happen 
        append!(all_V, ones(Tv, length(upper_tri_AV)))
        total_nnz += length(upper_tri_AV)
    end

    # nnz of sum_A correspond to potential nnz of 
    # \sum_{i=1}^m y_i A_i
    # thus via preallocation, we can speed up 
    # the computation of addition of sparse matrices
    sum_A = sparse(all_I, all_J, all_V, n, n)
    agg_A_ptr = zeros(Ti, nA + 1)
    agg_A_ind = zeros(Ti, total_nnz)
    agg_A_valone = zeros(Tv, total_nnz)
    agg_A_valtwo = zeros(Tv, total_nnz)

    cumul_nnz = 0
    for i in eachindex(As)
        # entries from agg_A_ptr[i] to agg_A_ptr[i+1]-1
        # correspond to the i-th sparse constraint/objective matrix
        agg_A_ptr[i] = cumul_nnz + 1
        upper_tri_I = upper_tri_I_list[i]
        upper_tri_J = upper_tri_J_list[i]
        upper_tri_V = upper_tri_V_list[i]
        for j in eachindex(upper_tri_I)
            row, col = upper_tri_I[j], upper_tri_J[j]
            low = sum_A.colptr[col]
            high = sum_A.colptr[col+1]-1
            while low <= high
                mid = (low + high) ÷ 2
                if sum_A.rowval[mid] == row 
                    agg_A_ind[cumul_nnz+j] = mid 
                    break
                elseif sum_A.rowval[mid] < row 
                    low = mid + 1
                else
                    high = mid - 1
                end
            end
            agg_A_valone[cumul_nnz+j] = upper_tri_V[j] 
            if row == col
                agg_A_valtwo[cumul_nnz+j] = upper_tri_V[j]
            else
                # since the matrix is symmetric, 
                # we can scale up the off-diagonal entries by 2
                agg_A_valtwo[cumul_nnz+j] = Tv(2.0) * upper_tri_V[j] 
            end
        end
    end
    agg_A_ptr[end] = total_nnz + 1
    return S, inds
end


"""
Interface for SDPLR 

Problem formulation:
    min   Tr(C (YY^T))
    s.t.  Tr(As[i] (YY^T)) = bs[i]
            Y in R^{n x r}
"""
function sdplr(
    C::AbstractMatrix{Tv},
    As::Vector{Any},
    b::Vector{Tv},
    r::Ti;
    config::BurerMonteiroConfig{Ti, Tv}=BurerMonteiroConfig{Ti, Tv}(),
) where{Ti <: Integer, Tv <: AbstractFloat}
    m = length(As)
    Constraints = Any[]
    sparsecons = SparseMatrixCSC{Tv, Ti}[]

    # treat diagonal matrices as sparse matrices
    if isa(C, SparseMatrixCSC) 
        push!(sparsecons, C)
    elseif isa(C, Diagonal)
        push!(sparsecons, sparse(C))
    end

    for A in As
        if isa(A, SparseMatrixCSC)
            push!(sparsecons, A)
        elseif isa(A, Diagonal)
            push!(sparsecons, sparse(A))
        end
    end

    S, inds = preprocess_sparsecons(sparsecons)
    cnt = 0

    fill!(S.nzval, zero(Tv))
    if isa(C, SparseMatrixCSC)
        cnt += 1
        obj = C
        indC = inds[cnt]
    elseif isa(C, Diagonal)
        cnt += 1
        obj = C
        indC = inds[cnt]
    elseif isa(C, LowRankMatrix)
        obj = LowRankMatrix(C.D, C.B, r) 
        indC = zeros(0)
    elseif isa(C, UnitLowRankMatrix)
        obj = UnitLowRankMatrix(C.B, r)
        indC = zeros(0)
    else
        obj = C
        indC = zeros(0)
    end

    indAs = Any[]
    for A in As
        if isa(A, LowRankMatrix)
            indA = zeros(0)
            push!(Constraints, LowRankMatrix(A.D, A.B, r))
        elseif isa(A, UnitLowRankMatrix)
            indA = zeros(0)
            push!(Constraints, UnitLowRankMatrix(A.B, r))
        elseif isa(A, SparseMatrixCSC)
            cnt += 1
            indA = inds[cnt]
            push!(Constraints, A)
        elseif isa(A, Diagonal)
            cnt += 1
            indA = inds[cnt]
            push!(Constraints, A)
        else
            indA = zeros(0)
            push!(Constraints, A)
        end
        push!(indAs, indA)
    end

    SDP = SDPProblem(m, Constraints, obj, b, S, indC, indAs)
    n = size(C, 1)
    R₀ = 2 .* rand(n, r) .- 1
    λ₀ = randn(m)
    BM = BurerMonteiro(
        R₀,              #R
        zeros(size(R₀)), #G, will be initialized later
        λ₀,         #λ
        zeros(m),         #vio(violation + obj), will be initialized later
        BurerMonterioMutableScalars(
            one(Tv) / n,          #σ
            zero(Tv),                #obj, will be initialized later
            time(),           #starttime
            zero(Tv),                #endtime 
            zero(Tv),                #time spent on computing dual bound
            zero(Tv),                #time spend on primal computation
        )
    )
    res = _sdplr(BM, SDP, config)
    return res 
end


function _sdplr(
    BM::BurerMonteiro{Tv},
    SDP::SDPProblem{Ti, Tv, TC, TCons},
    config::BurerMonteiroConfig{Ti, Tv},
) where{Ti <: Integer, Tv <: AbstractFloat, TC <: AbstractMatrix{Tv}, TCons}
    # misc declarations
    recalcfreq = 5 
    recalc_cnt = 10^7 
    difficulty = 3 
    bestinfeas = 1.0e10
    BM.scalars.starttime = time()
    lastprint = BM.scalars.starttime # timestamp of last print
    R₀ = deepcopy(BM.R) 
    λ₀ = deepcopy(BM.λ)

    # TODO setup printing
    if config.printlevel > 0
        printheading(1)
    end


    # set up algorithm parameters
    normb = norm(SDP.b, Inf)
    normC = norm(SDP.C, Inf)
    best_dualbd = -1.0e20

    # initialize lbfgs datastructures
    lbfgshis = LBFGSHistory{Ti, Tv}(
        config.numlbfgsvecs,
        LBFGSVector{Tv}[],
        Ref(config.numlbfgsvecs))

    for i = 1:config.numlbfgsvecs
        push!(lbfgshis.vecs, 
            LBFGSVector(zeros(Tv, size(BM.R)),
                     zeros(Tv, size(BM.R)),
                     Ref(zero(Tv)), Ref(zero(Tv))))
    end


    # TODO essential_calc
    tol_stationarity = config.tol_stationarity / BM.scalars.σ 
    #@show tol_stationarity
    𝓛_val, stationarity , primal_vio = 
        essential_calcs!(BM, SDP, normC, normb)
    majoriter = 0 
    iter = 0 # total number of iterations

    # save initial function value, notice that
    # here the constraints may not be satisfied,
    # which means the value may be smaller than the optimum
    origval = 𝓛_val 

    majoriter_end = false
    dir = similar(BM.R)

    while majoriter < config.maxmajoriter 
        #avoid goto in C
        current_majoriter_end = false
        λ_update = 0
        while ((config.σ_strategy == 0 && λ_update < λ_updatect)
            ||(config.σ_strategy != 0 && difficulty != 1)) 

            # increase lambda counter, reset local iter counter and lastval
            λ_update += 1
            localiter = 0
            lastval = 1.0e10

            # check stopping criteria: rho_c_val = norm of gradient
            # once stationarity condition is satisfied, then break
            if stationarity <= tol_stationarity 
                break
            end

            # in the local iteration, we keep optimizing
            # the subproblem using lbfgsb and return a solution
            # satisfying stationarity condition
            while (stationarity > tol_stationarity) 
                #@show tol_stationarity
                #increase both iter and localiter counters
                iter += 1
                localiter += 1
                # direction has been negated
                dirlbfgs_dt = @elapsed begin
                    dirlbfgs!(dir, BM, lbfgshis, negate=true)
                end
                @show dirlbfgs_dt
                #@show norm(dir)

                descent = dot(dir, BM.G)
                if isnan(descent) || descent >= 0 # not a descent direction
                    dir .= -BM.G # reverse back to gradient direction
                end

                lastval = 𝓛_val
                linesearch_dt = @elapsed begin
                    α, 𝓛_val = linesearch!(BM, SDP, dir, α_max=1.0, update=true) 
                end
                @show linesearch_dt
                #@printf("iter %d, 𝓛_val %.10lf α %.10lf\n", iter, 𝓛_val, α) 
                #@show iter, 𝓛_val

                @. BM.R += α * dir
                if recalc_cnt == 0
                    𝓛_val, stationarity, primal_vio = 
                        essential_calcs!(BM, SDP, normC, normb)
                    recalc_cnt = recalcfreq
                    #@show 𝓛_val, stationarity, primal_vio
                else
                    gradient!(BM, SDP)
                    stationarity = norm(BM.G, 2) / (1.0 + normC)
                    primal_vio = norm(BM.primal_vio, 2) / (1.0 + normb)
                    recalc_cnt -= 1
                end

                lbfgs_postprecess_dt = @elapsed begin
                    if config.numlbfgsvecs > 0 
                        lbfgs_postprocess!(BM, lbfgshis, dir, α)
                    end
                end
                @show lbfgs_postprecess_dt

                current_time = time() 
                if current_time - lastprint >= config.printfreq
                    lastprint = current_time
                    if config.printlevel > 0
                        printintermediate(majoriter, localiter, iter, 𝓛_val, 
                                  BM.obj, stationarity, primal_vio, best_dualbd)
                    end
                end   

                totaltime = time() - BM.scalars.starttime

                if (totaltime >= config.timelim 
                    || primal_vio <= config.tol_primal_vio
                    ||  iter >= 10^7)
                    @. BM.λ -= BM.scalars.σ * BM.primal_vio
                    current_majoriter_end = true
                    break
                end
                bestinfeas = min(primal_vio, bestinfeas)
            end

            if current_majoriter_end
                majoriter_end = true
                break
            end

            # update Lagrange multipliers and recalculate essentials
            @. BM.λ -= BM.scalars.σ * BM.primal_vio
            𝓛_val, stationarity, primal_vio = 
                essential_calcs!(BM, SDP, normC, normb)

            if config.σ_strategy == 1
                if localiter <= 10
                    difficulty = 1 # EASY
                elseif localiter > 10 && localiter <= 50 
                    difficulty = 2 # MEDIUM
                else
                    difficulty = 3 # HARD
                end
            end
        end # end one major iteration

        # TODO check dual bounds

        # cannot further improve infeasibility,
        # in other words to make the solution feasible, 
        # we get a ridiculously large value
        if 𝓛_val > 1.0e10 * abs(origval) 
            majoriter_end = true
            printf("Cannot reduce infeasibility any further.\n")
            break
        end

        if isnan(𝓛_val)
            println("Error(sdplrlib): Got NaN.")
            return 0
        end

        if majoriter_end
            break
        end

        # TODO potential rank reduction 

        # update sigma
        while true
            BM.scalars.σ *= config.σ_fac
            𝓛_val, stationarity, primal_vio = 
                essential_calcs!(BM, SDP, normC, normb)
            tol_stationarity = config.tol_stationarity / BM.scalars.σ
            if tol_stationarity < stationarity 
                break
            end
        end
        # refresh some parameters
        λ_update = 0
        if config.σ_strategy == 1
            difficulty = 3
        end

        majoriter += 1

        # clear bfgs vectors
        for i = 1:lbfgshis.m
            lbfgshis.vecs[i] = LBFGSVector(zeros(size(BM.R)), zeros(size(BM.R)), Ref(0.0), Ref(0.0))
        end
    end
    𝓛_val, stationarity, primal_vio = essential_calcs!(BM, SDP, normC, normb)
    println("Done")
    if config.checkdual
        BM.scalars.dual_time = @elapsed best_dualbd = dualbound(BM, SDP)
    end
    BM.scalars.endtime = time()
    totaltime = BM.scalars.endtime - BM.scalars.starttime
    BM.scalars.primal_time = totaltime - BM.scalars.dual_time
    return Dict([
        "R" => BM.R,
        "λ" => BM.λ,
        "R₀" => R₀,
        "λ₀" => λ₀,
        "stationarity" => stationarity,
        "primal_vio" => primal_vio,
        "obj" => BM.scalars.obj,
        "dualbd" => best_dualbd,
        "totattime" => totaltime,
        "dualtime" => BM.scalars.dual_time,
        "primaltime" => BM.scalars.primal_time,
    ])
end


function dualbound(
    BM::BurerMonteiro,
    SDP::SDPProblem, 
)
    n = size(BM.R, 1)
    op = ArpackSimpleFunctionOp(
        (y, x) -> begin
                mul!(y, SDP.C, x)
                for i = 1:SDP.m
                     y .-= BM.λ[i] * (SDP.constraints[i] * x) 
                end
                return y
        end, n)
    eigenvals, eigenvecs = symeigs(op, 1; which=:SA, ncv=min(100, n), maxiter=1000000)
    dualbound = real.(eigenvals[1]) 
    return dualbound 
end

