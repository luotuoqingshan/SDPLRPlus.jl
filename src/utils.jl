"""
    barvinok_pataki(n, m)

Compute the barvinok-pataki bound min{n, sqrt{2m} + 1}
for a SDP problem where X has size n x n and there are m constraints.
"""
function barvinok_pataki(n::T, m::T) where {T <: Int}
    return min(n, T(floor(sqrt(2*m)+1)))
end


#TODO: finish it
function goemans_williamson_random_rounding(
    A::SparseMatrixCSC{Tv, Ti},
    R::Matrix{Tv}
) where {Tv <: AbstractFloat, Ti <: Int}
    n, r = size(R)
    v = randn()
end


"""
    save_benchmark_results(res, filepath)

save benchmark results to a file.
"""
function save_benchmark_results(
    res,
    filepath::String,
)
    io = IOBuffer()
    show(io, "text/plain", res)
    s = String(take!(io))
    open(filepath, "w") do fid
        write(fid, s)
    end
end


"""
    save_profile_results

save the output of Profile to a file.
"""
function save_profile_results(
    filepath::String
)
    open(filepath, "w") do f
        Profile.print(IOContext(f, :displaysize => (24, 500)), combine=true)
    end
end