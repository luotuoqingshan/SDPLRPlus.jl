include("structures.jl")


"""
This function computes the augmented Lagrangian value
"""
function lagrangval!(
    pdata::ProblemData, 
    algdata::AlgorithmData, 
    R::Matrix{Float64}
    )
    calA = Aoper(pdata, R, R; same=1, obj=1)
    algdata.vio[1:pdata.m] = calA
    algdata.vio[1:pdata.m] -= pdata.bs 
    return algdata.vio[end] - algdata.lambda' * algdata.vio[1:pdata.m]
           + 0.5 * algdata.sigma * norm(algdata.vio[1:pdata.m], 2)^2
end


"""
This function computes the violation of constraints,
i.e. it computes \\cal{A}((UV^T + VU^T)/2)

same : 1 if U and V are the same matrix
     : 0 if U and V are different matrices
obj  : whether to compute the objective function value
large : 1 if the size of UV^T is large such that we cannot directly compute UV^T 
      : 0 if the size of UV^T is small, we simply compute UV^T
"""
function Aoper(
    pdata::ProblemData,
    U::Matrix{Float64},
    V::Matrix{Float64};
    same::Int=1,
    obj::Int=1,
    large::Int=0)

    # deal with sparse and diagonal constraints first
    base = 0
    # store results of \\cal{A}(UV^T + VU^T)/2
    calA = zeros(pdata.m + 1)
    if large
        # when the size of UV^T is large, we cannot afford to compute UV^T
        if same 
            # sparse constraints, Tr(AUU^T) = U .* (AU) 
            for i = 1:pdata.m_sp
                calA[i + base] = (pdata.A_sp[i] * U) .* U
            end
            base += pdata.m_sp
            # diagonal constraints, Tr(DUU^T) = U .* (D * U)
            for i = 1:pdata.m_d
                calA[i + base] = (pdata.A_d[i] * U) .* U 
            end

            base += pdata.m_d
        else
            for i = 1:pdata.m_sp
                calA[i + base] = (((pdata.A_sp[i] * U) .* V) + 
                    ((pdata.A_sp[i] * V) .* U)) / 2.0
            end
            base += pdata.m_sp
            for i = 1:pdata.m_d
                calA[i + base] = (((pdata.A_d[i] * U) .* V) + 
                    ((pdata.A_d[i] * V) .* U)) / 2.0
            end
            base += pdata.m_d
        end
    else
        # when the size of UV^T is small, we can directly compute UV^T,
        # which saves a lot of computation especially when the number of
        # constraints is large 
        if same
            UVt = U * U'
        else
            UVt = (U * V' + V * U') / 2.0
        for i = 1:pdata.m_sp
            calA[i + base] = pdata.A_sp[i] * UVt
        end 
        base += pdata.m_sp
        for i = 1:pdata.m_d
            calA[i + base] = pdata.A_d[i] * UVt
        end
        base += pdata.m_d
    end

    # deal with low-rank constraints second
    for i = 1:pdata.m_lr
        if same
            M = pdata.A_lr[i].B' * U
            calA[i] = (pdata.A_lr[i].D * M) .* M
        else
            M = pdata.A_lr[i].B' * U
            N = pdata.A_lr[i].B' * V
            calA[i] = (((pdata.A_lr[i].D * M) .* N) + 
                ((pdata.A_lr[i].D * N) .* M)) / 2.0
        end
    end
    # if obj = 1, deal with objective function value
    if obj == 1
        if same
            if typeof(pdata.C) <: Diagonal
                calA[end] = (pdata.C * U) .* U 
            elseif typeof(pdata.C) <: LowRankMatrix
                M = pdata.C.B' * U 
                calA[end] = (pdata.C.D * M) .* M 
            else
                calA[end] = (pdata.C * U) .* U 
            end
        else
            if typeof(pdata.C) <: Diagonal
                calA[end] = (((pdata.C * U) .* V) +
                    ((pdata.C * V) .* U)) / 2.0 
            elseif typeof(pdata.C) <: LowRankMatrix
                M = pdata.C.B' * U 
                N = pdata.C.B' * V
                calA[end] = ((pdata.C.D * M) .* N + 
                    (pdata.C.D * N) .* M) / 2.0 
            else
                calA[end] = (((pdata.C * U) .* V) +
                    ((pdata.C * V) .* U)) / 2.0
            end
        end
    end
    return calA
end


"""
This function computes the gradient of the augmented Lagrangian
"""
function gradient!(
    pdata::ProblemData,
    algdata::AlgorithmData,
    R::Matrix,
)
    m = pdata.m
    y = -(algdata.lambda - algdata.sigma * algdata.vio[1:m])

    algdata.G = pdata.C * R 

    for i = 1:m
        if i <= pdata.m_sp
            algdata.G += y[i] * pdata.A_sp[i] * R
        elseif i <= pdata.m_sp + pdata.m_d
            j = i - pdata.m_sp
            algdata.G += y[i] * pdata.A_d[j] * R
        else
            j = i - pdata.m_sp - pdata.m_d 
            algdata.G += y[i] * pdata.A_lr[j].B * 
                (pdata.A_lr[j].D * 
                (pdata.A_lr[j].B' * R))
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
            return norm(pdata.C.diag, "fro")
        elseif typeof(pdata.C) <: LowRankMatrix
            return fro_norm_lr(pdata.C)
        elseif typeof(pdata.C) <: SparseMatrixCSC
            return norm(pdata.C, "fro")
        end
    else
        if matnum <= pdata.m_sp
            return norm(pdata.A_sp[matnum], "fro") 
        elseif matnum <= pdata.m_sp + pdata.m_d
            return norm(pdata.A_d[matnum - pdata.m_sp], "fro")
        else
            return fro_norm_lr(pdata.A_lr[matnum - pdata.m_sp - pdata.m_d])
    end
end


function essential_calcs!(
    pdata::ProblemData,
    algdata::AlgorithmData,
    R::Matrix{Float64},
    normC::Float64,
    normb::Float64,
)
    val = lagrangval!(pdata, algdata, R)
    gradient!(pdata, algdata, R)
    rho_c_val = norm(G, "fro") / (1.0 + normC)
    rho_f_val = norm(algdata.vio, 2) / (1.0 + normb)
    return (val, rho_c_val, rho_f_val)
end
