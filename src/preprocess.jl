"""
Extract upper triangular part of a sparse matrix in COO format.
"""
function LinearAlgebra.triu(coo::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    I = Ti[]; J = Ti[]; V = Tv[];
    for (i, j, v) in zip(coo.is, coo.js, coo.vs)
        if i <= j
            push!(I, i); push!(J, j); push!(V, v)
        end
    end
    return SparseMatrixCOO(I, J, V, coo.m, coo.n);
end


"""
    preprocess_sparsecons(As)

Preprocess the sparse constraints to initialize necessary 
data structures for BurerMonteiro algorithm.
"""
function preprocess_sparsecons(
    As::Vector{Union{SparseMatrixCSC{Tv, Ti}, SparseMatrixCOO{Tv, Ti}}}
) where {Ti <: Integer, Tv}
    # Following the original SDPLR code, we preprocess all the sparse 
    # constraints. Because the original code has limited documentation 
    # for this part, I add some explanation here. 


    # According to my understanding, this preprocess has two benefits, 
    # 1. It may increase the data locality, but this one is not benchmarked 
    # and the effects on many SDPs should be negligible. Maybe the locality 
    # is even worse.  
    # 2. A great part of the SDP solving involves evaluating the constraints/objective,
    # which is computing many frobenius inner products between symmetric 
    # matrices, this part can be done efficiently with only considering 
    # the upper triangular part. However, a full A is still required 
    # when computing the gradient. That's the reason we have two sets 
    # of auxiliary variables and also the mapping between them. 


    # aggregate all constraints into one sparse matrix
    n = size(As[1], 1)
    nA = length(As)

    # first count the nnz to perform preallocation
    total_nnz = 0; total_triu_nnz = 0
    for A in As
        total_nnz += nnz(A)
        total_triu_nnz += nnz(triu(A))
    end
    
    # (all_triu_I, all_triu_J, all_triu_V) stores 
    # combined upper triangular part of all constraint matrices A
    all_triu_I = zeros(Ti, total_triu_nnz)
    all_triu_J = zeros(Ti, total_triu_nnz)
    all_triu_V = ones(Tv, total_triu_nnz)

    # (all_I, all_J, all_V) stores
    # combined entries of all constraint matrices A
    all_I = zeros(Ti, total_nnz)
    all_J = zeros(Ti, total_nnz)
    all_V = ones(Tv, total_nnz)

    cur_ptr = 0
    cur_triu_ptr = 0

    for A in As
        # extract upper triangular part of A
        I, J, _ = findnz(A)
        triu_I, triu_J, _ = findnz(triu(A))

        # here we append 1 instead of V[i] to both all_triu_V and all_V
        # in this way we can make sure cancellation doesn't happen 
        all_I[cur_ptr+1:cur_ptr+length(I)] .= I
        all_J[cur_ptr+1:cur_ptr+length(J)] .= J
        all_triu_I[cur_triu_ptr+1:cur_triu_ptr+length(triu_I)] .= triu_I
        all_triu_J[cur_triu_ptr+1:cur_triu_ptr+length(triu_J)] .= triu_J
        cur_ptr += length(I)
        cur_triu_ptr += length(triu_I)
    end

    # nnz of agg_sparse_A correspond to potential nnz of 
    # \sum_{i=1}^m y_i A_i
    # thus via preallocation, we can speed up 
    # the computation of addition of sparse matrices

    # combine all upper triangular parts of As
    triu_agg_sparse_A = sparse(all_triu_I, all_triu_J, all_triu_V, n, n)

    # combine all entries of As
    agg_sparse_A = sparse(all_I, all_J, all_V, n, n)

    triu_agg_sparse_A_matptr = zeros(Ti, nA + 1)
    triu_agg_sparse_A_nzind = zeros(Ti, total_triu_nnz)
    triu_agg_sparse_A_nzval_one = zeros(Tv, total_triu_nnz)
    triu_agg_sparse_A_nzval_two = zeros(Tv, total_triu_nnz)

    cumul_nnz = 0
    for i in eachindex(As)
        # entries from triu_agg_sparse_A_matptr[i] to triu_agg_sparse_A_matptr[i+1]-1
        # correspond to the i-th sparse constraint/objective matrix
        triu_agg_sparse_A_matptr[i] = cumul_nnz + 1
        triu_I, triu_J, triu_V = findnz(triu(As[i]))
        for j in eachindex(triu_I)
            row, col = triu_I[j], triu_J[j]
            low = triu_agg_sparse_A.colptr[col]
            high = triu_agg_sparse_A.colptr[col+1]-1
            # binary search where the jth entry of triu(As[i])
            # is located in triu_agg_sparse_A
            while low <= high
                mid = (low + high) ÷ 2
                if triu_agg_sparse_A.rowval[mid] == row 
                    triu_agg_sparse_A_nzind[cumul_nnz+j] = mid 
                    break
                elseif triu_agg_sparse_A.rowval[mid] < row 
                    low = mid + 1
                else
                    high = mid - 1
                end
            end
            triu_agg_sparse_A_nzval_one[cumul_nnz+j] = triu_V[j] 
            if row == col
                triu_agg_sparse_A_nzval_two[cumul_nnz+j] = triu_V[j]
            else
                # since the matrix is symmetric, 
                # we can scale up the off-diagonal entries by 2
                # as we are considering upper triangular part
                triu_agg_sparse_A_nzval_two[cumul_nnz+j] = Tv(2.0) * triu_V[j] 
            end
        end
        cumul_nnz += length(triu_I)
    end
    triu_agg_sparse_A_matptr[end] = total_triu_nnz + 1

    # map the entries of agg_sparse_A to the corresponding entries of triu_agg_sparse_A
    agg_sparse_A_mappedto_triu = zeros(Ti, length(agg_sparse_A.rowval))
    for col = 1:n
        for nzi = agg_sparse_A.colptr[col]:agg_sparse_A.colptr[col+1]-1
            row = agg_sparse_A.rowval[nzi]
            r = min(row, col)
            c = max(row, col)
            low = triu_agg_sparse_A.colptr[c]
            high = triu_agg_sparse_A.colptr[c+1]-1
            #@show low, high, triu_sum_A.rowval[low:high]
            while low <= high
                mid = div(low + high, 2)
                if triu_agg_sparse_A.rowval[mid] == r
                    agg_sparse_A_mappedto_triu[nzi] = mid
                    break
                elseif triu_agg_sparse_A.rowval[mid] > r
                    high = mid - 1
                else
                    low = mid + 1
                end
            end
        end
    end
    return (triu_agg_sparse_A, triu_agg_sparse_A_matptr, 
            triu_agg_sparse_A_nzind, triu_agg_sparse_A_nzval_one, 
            triu_agg_sparse_A_nzval_two, agg_sparse_A, 
            agg_sparse_A_mappedto_triu)
end