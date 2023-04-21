include("structures.jl")


"""
This function computes the augmented Lagrangian value
"""
function lagrangval!(
    pdata::ProblemData, 
    algdata::AlgorithmData, 
    )
    # apply the operator 𝓐 to RRᵀ and 
    # potentially compute the objective function value
    obj, calA = Aoper(pdata, algdata.R, algdata.R; same=true, calcobj=true)
    algdata.vio = calA - pdata.b 
    algdata.obj = obj
    return algdata.obj - algdata.λ' * algdata.vio
           + 0.5 * algdata.σ * norm(algdata.vio, 2)^2
end


"""
This function computes the violation of constraints,
i.e. it computes 𝓐((UVᵀ + VUᵀ)/2)

same : 1 if U and V are the same matrix
     : 0 if U and V are different matrices
obj  : whether to compute the objective function value
large : 1 if the size of UVᵀ is large such that we cannot directly compute UVᵀ 
      : 0 if the size of UVᵀ is small, we simply compute UVᵀ
"""
function Aoper(
    pdata::ProblemData,
    U::Matrix{Float64},
    V::Matrix{Float64};
    same::Bool=true,
    calcobj::Bool=true,
    large::Bool=false,
)
    # deal with sparse and diagonal constraints first
    base = 0
    # store results of 𝓐(UVᵀ + VUᵀ)/2
    calA = zeros(pdata.m)
    if large 
        # when the size of UVᵀ is large, we cannot afford to compute UVᵀ
        if same  
            # sparse constraints, Tr(AUUᵀ) = sum(U .* (AU))
            for i = 1:pdata.m_sp
                calA[i + base] = sum((pdata.A_sp[i] * U) .* U)
            end
            base += pdata.m_sp
            # diagonal constraints, Tr(DUUᵀ) = sum(U .* (D * U))
            for i = 1:pdata.m_diag
                calA[i + base] = sum((pdata.A_diag[i] * U) .* U)
            end

            base += pdata.m_diag
        else
            for i = 1:pdata.m_sp
                calA[i + base] = sum(((pdata.A_sp[i] * U) .* V) + 
                    ((pdata.A_sp[i] * V) .* U)) / 2.0
            end
            base += pdata.m_sp
            for i = 1:pdata.m_diag
                calA[i + base] = sum(((pdata.A_diag[i] * U) .* V) + 
                    ((pdata.A_diag[i] * V) .* U)) / 2.0
            end
            base += pdata.m_diag
        end
    else
        # when the size of UVᵀ is small, we can directly compute UVᵀ,
        # which saves a lot of computation especially when the number of
        # constraints is large 
        if same 
            UVt = U * U'
        else
            UVt = (U * V' + V * U') / 2.0
        end
        for i = 1:pdata.m_sp
            calA[i + base] = sum(pdata.A_sp[i] * UVt)
        end 
        base += pdata.m_sp
        for i = 1:pdata.m_diag
            calA[i + base] = sum(pdata.A_diag[i] * UVt)
        end
        base += pdata.m_diag
    end

    # deal with low-rank constraints second
    for i = 1:pdata.m_lr
        if same 
            M = pdata.A_lr[i].B' * U
            calA[i] = sum((pdata.A_lr[i].D * M) .* M)
        else
            M = pdata.A_lr[i].B' * U
            N = pdata.A_lr[i].B' * V
            calA[i] = sum(((pdata.A_lr[i].D * M) .* N) + 
                ((pdata.A_lr[i].D * N) .* M)) / 2.0
        end
    end

    #deal with dense constraints last
    for i = 1:pdata.m_dense
        if same
        end
    end
    # if calcobj = true, deal with objective function value
    obj = 0.0
    if calcobj 
        if same
            if typeof(pdata.C) <: Diagonal
                obj = sum((pdata.C * U) .* U) 
            elseif typeof(pdata.C) <: LowRankMatrix
                M = pdata.C.B' * U 
                obj = sum((pdata.C.D * M) .* M)
            else
                obj = sum((pdata.C * U) .* U)
            end
        else
            if typeof(pdata.C) <: Diagonal
                obj = sum(((pdata.C * U) .* V) +
                    ((pdata.C * V) .* U)) / 2.0 
            elseif typeof(pdata.C) <: LowRankMatrix
                M = pdata.C.B' * U 
                N = pdata.C.B' * V
                obj = sum((pdata.C.D * M) .* N + 
                    (pdata.C.D * N) .* M) / 2.0 
            else
                obj = sum(((pdata.C * U) .* V) +
                    ((pdata.C * V) .* U)) / 2.0
            end
        end
    end
    return (obj, calA)
end


"""
This function computes the gradient of the augmented Lagrangian
"""
function gradient!(
    pdata::ProblemData,
    algdata::AlgorithmData,
)
    m = pdata.m
    y = -(algdata.λ - algdata.σ * algdata.vio)

    algdata.G = pdata.C * algdata.R 

    for i = 1:m
        if i <= pdata.m_sp
            algdata.G += y[i] * pdata.A_sp[i] * algdata.R
        elseif i <= pdata.m_sp + pdata.m_d
            j = i - pdata.m_sp
            algdata.G += y[i] * pdata.A_d[j] * algdata.R
        else
            j = i - pdata.m_sp - pdata.m_d 
            algdata.G += y[i] * pdata.A_lr[j].B * 
                (pdata.A_lr[j].D * 
                (pdata.A_lr[j].B' * algdata.R))
        end
    end
    algdata.G .*= 2.0
    return 0
end


"""
Compute the maximum magnitude of
elements in the objective matrix C
"""
function C_normdatamat(
    pdata::ProblemData
)
    Cnorm = 0.0
    if typeof(pdata.C) <: Diagonal
        Cnorm = norm(pdata.C.diag, Inf) 
    elseif typeof(pdata.C) <: LowRankMatrix
        # C = BDB^T
        U = pdata.C.B * pdata.C.D
        for i = axes(U, 1) 
            Cnorm = max(Cnorm, norm(U[i, :] * pdata.C.B', Inf))
        end
    elseif typeof(pdata.C) <: SparseMatrixCSC
        Cnorm = maximum(abs.(pdata.C))
    end
    return Cnorm
end


"""
Compute Frobenius norm for low-rank matrix A = B * D * B^T
where B is tall rectangular matrix.
"""
function fro_norm_lr(
    A::LowRankMatrix,
)
    # A = BDB^T
    # ||A||_F^2 = Tr(A^T A) = Tr(B D B^TB DB^T) = Tr((B^T B)D (B^T B)D)
    C = (A.B' * A.B) * A.D
    return sqrt(tr(C * C))
end


"""
Compute Frobenius norm of the data matrix A or objective matrix C
"""
function normdatamat(
    pdata::ProblemData,
    matnum::Int,
)
    # matnum = 0 : objective matrix
    # matnum > 0 : constraint matrix
    @assert matnum <= pdata.m && matnum >= 0
    if matnum == 0 # objective matrix
        if typeof(pdata.C) <: Diagonal
            return norm(pdata.C.diag, 2)
        elseif typeof(pdata.C) <: LowRankMatrix
            return fro_norm_lr(pdata.C)
        elseif typeof(pdata.C) <: SparseMatrixCSC
            return norm(pdata.C, 2)
        end
    else
        if matnum <= pdata.m_sp
            return norm(pdata.A_sp[matnum], 2) 
        elseif matnum <= pdata.m_sp + pdata.m_d
            return norm(pdata.A_d[matnum - pdata.m_sp], 2)
        else
            return fro_norm_lr(pdata.A_lr[matnum - pdata.m_sp - pdata.m_d])
        end
    end
end


function essential_calcs!(
    pdata::ProblemData,
    algdata::AlgorithmData,
    normC::Float64,
    normb::Float64,
)
    val = lagrangval!(pdata, algdata)
    gradient!(pdata, algdata)
    ρ_c_val = norm(algdata.G, 2) / (1.0 + normC)
    ρ_f_val = norm(algdata.vio, 2) / (1.0 + normb)
    return (val, ρ_c_val, ρ_f_val)
end
