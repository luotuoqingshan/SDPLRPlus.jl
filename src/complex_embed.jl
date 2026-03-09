# Real embedding for complex Hermitian SDPs.
#
# For a complex Hermitian n×n matrix A, the real embedding is the 2n×2n real
# symmetric matrix:
#
#   phi(A) = [ Re(A)   -Im(A) ]
#             [ Im(A)    Re(A) ]
#
# Key property:  tr(phi(A)/2 * phi(X)) = Re(tr(A * X))
#
# By dividing all embedded matrices by 2 the inner product is preserved exactly,
# so the real 2n×2n SDP is equivalent to the original complex Hermitian SDP.
# This lets us pass complex SDPs through the real-valued solver without any
# changes to the core algorithm.

"""
    embed_matrix(A)

Embed a complex Hermitian n×n matrix into a real symmetric 2n×2n matrix,
scaled by 1/2 so that the inner product is preserved:

    tr(embed_matrix(A) * phi(X)) = Re(tr(A * X))

where phi(X) = [Re(X) -Im(X); Im(X) Re(X)].
"""
function embed_matrix(A::SparseMatrixCSC{Tv,Ti}) where {Tv<:Complex,Ti<:Integer}
    n = size(A, 1)
    rows_A, cols_A, vals_A = findnz(A)
    nnz_A = length(vals_A)
    Tr = real(Tv)
    I = Vector{Int}(undef, 4 * nnz_A)
    J = Vector{Int}(undef, 4 * nnz_A)
    V = Vector{Tr}(undef, 4 * nnz_A)
    @inbounds for k in 1:nnz_A
        i, j, v = Int(rows_A[k]), Int(cols_A[k]), vals_A[k]
        re_v = real(v) / 2
        im_v = imag(v) / 2
        # Top-left block:    Re(A)/2
        I[k] = i
        J[k] = j
        V[k] = re_v
        # Top-right block:  -Im(A)/2
        I[k+nnz_A] = i
        J[k+nnz_A] = j + n
        V[k+nnz_A] = -im_v
        # Bottom-left block: Im(A)/2
        I[k+2*nnz_A] = i + n
        J[k+2*nnz_A] = j
        V[k+2*nnz_A] = im_v
        # Bottom-right block: Re(A)/2
        I[k+3*nnz_A] = i + n
        J[k+3*nnz_A] = j + n
        V[k+3*nnz_A] = re_v
    end
    return sparse(I, J, V, 2 * n, 2 * n)
end

function embed_matrix(A::Diagonal{Tv}) where {Tv<:Complex}
    # A Hermitian diagonal must have real entries (D_ii = conj(D_ii) => Im = 0).
    d = real.(A.diag) ./ 2
    return Diagonal(vcat(d, d))
end

function embed_matrix(A::SymLowRankMatrix{Tv}) where {Tv<:Complex}
    # SymLowRankMatrix(D, B) with complex B represents B * D * B^H because
    # Julia's B' is the conjugate transpose.  The real embedding is:
    #   phi(B D B^H)/2 = phi_B * Diagonal(vcat(d,d)/2) * phi_B^T
    # where phi_B = [Re(B) -Im(B); Im(B) Re(B)]  (2n x 2k, real).
    B = A.B                         # n x k complex
    d = real.(A.D.diag)             # k-vector (eigenvalues, real for Hermitian)
    Re_B = real.(B)
    Im_B = imag.(B)
    phi_B = [Re_B -Im_B; Im_B Re_B]  # (2n) x (2k) real
    D_tilde = Diagonal(vcat(d, d) ./ 2)
    return SymLowRankMatrix(D_tilde, phi_B)
end

function embed_matrix(A::SparseMatrixCOO{Tv,Ti}) where {Tv<:Complex,Ti<:Integer}
    # Convert to CSC then embed.
    return embed_matrix(sparse(A.is, A.js, A.vs, size(A, 1), size(A, 2)))
end

"""
    embed_hermitian_sdp(C, As, b)

Transform a complex Hermitian SDP into an equivalent real symmetric SDP
via the 2n×2n real embedding.

Returns `(C_real, As_real, b_real)` where:
- `C_real` is a 2n×2n real matrix (= phi(C)/2)
- `As_real[i]` is a 2n×2n real matrix (= phi(As[i])/2)
- `b_real` is a real copy of `b` (unchanged; the factor-of-2 from the trace
  identity is absorbed into the /2 scaling of the matrices)

The returned real SDP has the same optimal value as the original complex SDP:

    min  Re(tr(C X))   s.t. Re(tr(As[i] X)) = b[i],  X Hermitian PSD
    <=>
    min  tr(C_real X') s.t.    tr(As_real[i] X') = b[i], X' symmetric PSD
"""
function embed_hermitian_sdp(C, As, b::AbstractVector)
    C_real = embed_matrix(C)
    As_real = [embed_matrix(A) for A in As]
    Tr = real(eltype(C_real))
    b_real = Vector{Tr}(real.(b))
    return C_real, As_real, b_real
end

"""
    extract_complex_result!(result, n_orig)

Post-process the result dictionary returned by the real embedded solver.

The real factor `Rt` has shape `(r_real, 2*n_orig)`.  The first `n_orig`
columns encode Re(R) and the next `n_orig` columns encode Im(R), so the
complex primal factor is:

    Rt_complex = Rt_real[:, 1:n_orig] + im * Rt_real[:, n_orig+1:end]

Modifies `result` in-place and returns it.
"""
function extract_complex_result!(result::Dict, n_orig::Integer)
    function to_complex(Rt_real)
        return Rt_real[:, 1:n_orig] .+ im .* Rt_real[:, (n_orig+1):end]
    end
    result["Rt"] = to_complex(result["Rt"])
    result["Rt0"] = to_complex(result["Rt0"])
    return result
end
