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


"""

Write initial solution which can be loaded into SDPLR-1.03-beta.
"""
function write_initial_solution(
    R::Matrix{Tv}, 
    λ::Vector{Tv}, 
    filename::String;
    filefolder::String=homedir()*"/SDPLR-1.03-beta 3/data/",
) where {Tv <: AbstractFloat}
    n, r = size(R)
    m = length(λ)
    filepath = filefolder*filename
    @show filepath
    open(filepath, "w") do f
        write(f, "dual variable $m\n")
        for i = 1:m
            write(f, "$(λ[i])\n")
        end
        write(f, "primal variable 1 s $n $r $r\n")
        for j = axes(R, 2) 
            for i = axes(R, 1)
                write(f, "$(R[i, j])\n")
            end
        end
        write(f, "special majiter 0\n")
        write(f, "special iter 0\n")
        write(f, "special lambdaupdate 0")
        write(f, "special CG 0\n")
        write(f, "special curr_CG 0\n")
        write(f, "special totaltime 0\n")
        write(f, "special sigma $(1.0/n)\n") 
        write(f, "special scale 1.0\n")
    end
    println("Finishing writing initial solution to $filepath")
end

include("optprograms.jl")

#for i = [70, 72, 77, 81] 
#    A = load_gset("G$i")
#    C, As, bs = maxcut(A);
#    write_problem_sdpa("G$i.sdpa", C, As, bs)
#end
