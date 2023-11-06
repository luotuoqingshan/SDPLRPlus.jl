
function barvinok_pataki(n::T, m::T) where {T <: Int}
    return min(n, T(floor(sqrt(2*m)+1)))
end


function goemans_williamson_random_rounding(
    A::SparseMatrixCSC{Tv, Ti},
    R::Matrix{Tv}
) where {Tv <: AbstractFloat, Ti <: Int}
    n, r = size(R)
    v = randn()
end


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