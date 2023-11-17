using DelimitedFiles
using SparseArrays, LinearAlgebra


"""
    load_gset(filename; [filepath])

Load a GSet format graph file into an adjacency matrix.
"""
function load_gset(
    filename::String;
    filefolder::String=homedir()*"/Gset/",
)::SparseMatrixCSC
    I = Int32[] 
    J = Int32[]
    V = Float64[]
    n = 0
    m = 0
    filepath = filefolder*filename
    open(filepath) do file
        lines = readlines(file)
        n, m = split(lines[1], ' ')
        n = parse(Int, n)
        m = parse(Int, m)
        for line in lines[2:end]
            u, v, w = split(line, ' ')
            u = parse(Int, u)
            v = parse(Int, v)
            w = parse(Float64, w)
            push!(I, u)
            push!(J, v)
            push!(V, w)
        end
    end
    A = sparse(I, J, V, n, n)
    # Turn it into an undirected graph
    A = max.(A, A')
    # remove self-loops
    A[diagind(A)] .= 0
    dropzeros!(A)
    return A
end


"""
    write_problem_sdpa(filename, C, As, bs; [filefolder])

Save the SDP problem in the SDPA format.
"""
function write_problem_sdpa(
    filename::String,
    C::AbstractMatrix{Tv},
    As::TCons,
    bs::Vector{Tv};
    filefolder::String=homedir()*"/SDPLR-1.03-beta 3/data/"
) where {Tv <: AbstractFloat, TCons}
    n = size(C, 1) 
    m = length(As)
    filepath = filefolder*filename
    open(filepath, "w") do f
        write(f, "$m\n") # number of constraint matrices
        write(f, "1\n")  # number of blocks in the SDP
        write(f, "$n\n") # sizes of the blocks
        for i = 1:m
            write(f, "$(bs[i]) ")
        end
        write(f, "\n")
        triuC = triu(C)
        for (i, j, v) in zip(findnz(triuC)...)
            write(f, "0 1 $i $j $(-v)\n")
        end
        for i = 1:m
            triuAi = triu(As[i])
            for (r, c, v) in zip(findnz(triuAi)...)
                write(f, "$i 1 $r $c $v\n")
            end
        end
    end
end

include("optprograms.jl")

A = load_gset("G1")

C, As, bs = maxcut(A);

write_problem_sdpa("G1.sdpa", C, As, bs)