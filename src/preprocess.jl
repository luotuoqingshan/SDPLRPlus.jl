"""
    sparse_sym_to_triu(I, J, V)

Extract the upper triangular part of a symmetric sparse matrix in the (I, J, V) format.
Both of the input and output are in the (I, J, V) format.
"""
function sparse_sym_to_triu(
    I::Vector{Ti}, 
    J::Vector{Ti},
    V::Vector{Tv},
)where{Ti <: Integer, Tv <: AbstractFloat}
    triu_I, triu_J, triu_V = Ti[], Ti[], Tv[]
    for i in eachindex(I)
        # only store upper triangular part
        if I[i] <= J[i] 
            push!(triu_I, I[i])
            push!(triu_J, J[i])
            push!(triu_V, V[i])
        end
    end
    return triu_I, triu_J, triu_V
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

    # (all_triu_I, all_triu_J, all_triu_V) stores 
    # combined upper triangular part of all constraint matrices A
    all_triu_I, all_triu_J, all_triu_V = Ti[], Ti[], Tv[]

    # (all_I, all_J, all_V) stores
    # combined entries of all constraint matrices A
    all_I, all_J, all_V = Ti[], Ti[], Tv[]

    n = size(As[1], 1)
    nA = length(As)
    total_nnz = 0

    # (triu_I_list, triu_J_list, triu_V_list) stores
    # upper triangular part of each constraint matrix A
    triu_I_list = Vector{Ti}[]
    triu_J_list = Vector{Ti}[]
    triu_V_list = Vector{Tv}[]

    for A in As
        # extract upper triangular part of A
        I, J, V = findnz(A)
        triu_I, triu_J, triu_V = 
            sparse_sym_to_triu(I, J, V)
        push!(triu_I_list, triu_I)
        push!(triu_J_list, triu_J)
        push!(triu_V_list, triu_V)

        # here we append 1 instead of V[i] to both all_triu_V and all_V
        # in this way we can make sure cancellation doesn't happen 
        append!(all_triu_I, triu_I)
        append!(all_triu_J, triu_J)
        append!(all_triu_V, ones(Tv, length(triu_V)))

        append!(all_I, I)
        append!(all_J, J)
        append!(all_V, ones(Tv, length(V)))
        total_nnz += length(triu_V)
    end

    # nnz of sum_A correspond to potential nnz of 
    # \sum_{i=1}^m y_i A_i
    # thus via preallocation, we can speed up 
    # the computation of addition of sparse matrices

    # combine all upper triangular parts of As
    triu_sum_A = sparse(all_triu_I, all_triu_J, all_triu_V, n, n)

    # combine all entries of As
    sum_A = sparse(all_I, all_J, all_V, n, n)

    agg_A_ptr = zeros(Ti, nA + 1)
    agg_A_nzind = zeros(Ti, total_nnz)
    agg_A_nzval_one = zeros(Tv, total_nnz)
    agg_A_nzval_two = zeros(Tv, total_nnz)

    cumul_nnz = 0
    for i in eachindex(As)
        # entries from agg_A_ptr[i] to agg_A_ptr[i+1]-1
        # correspond to the i-th sparse constraint/objective matrix
        agg_A_ptr[i] = cumul_nnz + 1
        triu_I = triu_I_list[i]
        triu_J = triu_J_list[i]
        triu_V = triu_V_list[i]
        for j in eachindex(triu_I)
            row, col = triu_I[j], triu_J[j]
            low = triu_sum_A.colptr[col]
            high = triu_sum_A.colptr[col+1]-1
            while low <= high
                mid = (low + high) ÷ 2
                if triu_sum_A.rowval[mid] == row 
                    agg_A_nzind[cumul_nnz+j] = mid 
                    break
                elseif triu_sum_A.rowval[mid] < row 
                    low = mid + 1
                else
                    high = mid - 1
                end
            end
            agg_A_nzval_one[cumul_nnz+j] = triu_V[j] 
            if row == col
                agg_A_nzval_two[cumul_nnz+j] = triu_V[j]
            else
                # since the matrix is symmetric, 
                # we can scale up the off-diagonal entries by 2
                agg_A_nzval_two[cumul_nnz+j] = Tv(2.0) * triu_V[j] 
            end
        end
        cumul_nnz += length(triu_I)
    end
    agg_A_ptr[end] = total_nnz + 1
    sum_A_triu_sum_A_inds = zeros(Ti, length(sum_A.rowval))
    for col = 1:n
        for nzi = sum_A.colptr[col]:sum_A.colptr[col+1]-1
            row = sum_A.rowval[nzi]
            r = min(row, col)
            c = max(row, col)
            low = triu_sum_A.colptr[c]
            high = triu_sum_A.colptr[c+1]-1
            #@show low, high, triu_sum_A.rowval[low:high]
            while low <= high
                mid = div(low + high, 2)
                if triu_sum_A.rowval[mid] == r
                    sum_A_triu_sum_A_inds[nzi] = mid
                    break
                elseif triu_sum_A.rowval[mid] > r
                    high = mid - 1
                else
                    low = mid + 1
                end
            end
        end
    end
    return (triu_sum_A, agg_A_ptr, agg_A_nzind, agg_A_nzval_one, 
           agg_A_nzval_two, sum_A, sum_A_triu_sum_A_inds)
end