using DelimitedFiles
using SparseArrays, LinearAlgebra

function load_gset(filepath)::SparseMatrixCSC
    I = Int32[] 
    J = Int32[]
    V = Float64[]
    n = 0
    m = 0
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
