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
            # Turn it into an undirected graph
            push!(I, u); push!(J, v); push!(V, w)
            push!(J, u); push!(I, v); push!(V, w)
        end
    end
    A = sparse(I, J, V, n, n)
    # remove self-loops
    A[diagind(A)] .= 0
    dropzeros!(A)
    return A
end


"""
    load_gset_smat(filename; [filepath])

Load a GSet graph in the smat format into an adjacency matrix.
"""
function load_gset_smat(
    filename::String;
    filefolder::String=homedir()*"/Gset/",
)
    I = Int32[] 
    J = Int32[]
    V = Float64[]
    n = 0
    m = 0
    filepath = filefolder*filename*".smat"
    open(filepath) do file
        lines = readlines(file)
        n, m, nnz_A = split(lines[1], ' ')
        n = parse(Int, n)
        m = parse(Int, m)
        for line in lines[2:end]
            u, v, w = split(line, ' ')
            u = parse(Int, u)
            v = parse(Int, v)
            w = parse(Float64, w)
            push!(I, u+1)
            push!(J, v+1)
            push!(V, w)
        end
    end
    A = sparse(I, J, V, n, m)
    # remove self-loops
    A[diagind(A)] .= 0
    dropzeros!(A)
    return A
end


"""
    write_problem_sdpa(filename, C, As, bs; [filefolder])

Save the SDP problem in the SDPA format.

Doc for SDPA: https://plato.asu.edu/ftp/sdpa_format.txt
"""
function write_problem_sdpa(
    filename::String,
    C::AbstractMatrix{Tv},
    As::TCons,
    bs::Vector{Tv};
    filefolder::String=homedir()*"/SDPLR-1.03-beta/data/"
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


function write_matrix_sdplr(
    A::SparseMatrixCSC{Tv, Ti},
    id::Ti, 
    f::IOStream,
) where {Tv <: AbstractFloat, Ti <: Integer}
    triuA = triu(A)
    write(f, "$id 1 s $(nnz(triuA))\n")
    for (i, j, v) in zip(findnz(triuA)...)
        write(f, "$i $j $v\n")
    end
end


function write_matrix_sdplr(
    A::LowRankMatrix{Tv},
    id::Ti,
    f::IOStream,
) where {Tv <: AbstractFloat, Ti <: Integer}
    # matrix id, block id, low rank, rank
    write(f, "$id 1 l $(size(A.B, 2))\n")
    # write down the diagonal
    for i = axes(A.D, 1)
        write(f, "$(A.D[i, i])\n")
    end
    # write down B in column major order
    for j = axes(A.B, 2)
        for i = axes(A.B, 1)
            write(f, "$(A.B[i, j])\n")
        end
    end
end


function write_matrix_sdplr(
    A::AbstractMatrix{Tv},
    id::Ti,
    f::IOStream,
) where {Tv <: AbstractFloat, Ti <: Integer}
    @error "Only sparse and low-rank matrices are supported in SDPLR."
end


"""
    write_problem_sdplr(filename, C, As, bs; [filefolder])

Save the SDP problem in the SDPLR format.

Doc for SDPLR: https://sburer.github.io/files/SDPLR-1.03-beta-usrguide.pdf 
"""
function write_problem_sdplr(
    filename::String,
    C::AbstractMatrix{Tv},
    As::TCons,
    bs::Vector{Tv};
    filefolder::String=homedir()*"/SDPLR-1.03-beta/data/"
) where {Tv <: AbstractFloat, TCons,}
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
        write(f, "1\n") # this line is currently ignored
        write_matrix_sdplr(C, 0, f)
        for i = 1:m
            write_matrix_sdplr(As[i], i, f)
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



"""
    gset([gset_folder, output_folder;program, sdp_format])

Batch process Gset graphs, write out their SDP programs,
and store them in the SDPA/SDPLR format. 
"""
function gset_sdp_preprocess_data(
    gset_folder::String=homedir()*"/Gset/",
    output_folder::String=homedir()*"/SDPLR-1.03-beta/data/";
    program="maxcut",
    sdp_format="sdpa",  
)
    for i = [1:67; 70; 72; 77; 81]
        A = load_gset("G$i"; filefolder=gset_folder)
        if program == "maxcut"
            C, As, bs = maxcut(A)
        elseif program == "lovasz_theta"
            C, As, bs = lovasz_theta(A)
        elseif program == "minimum_bisection"
            C, As, bs = minimum_bisection(A)
        else
            @error "Currently we only support maxcut/lovasz_theta/minimum_bisection."
        end
        if sdp_format == "sdpa"
            write_problem_sdpa("G$i."*sdp_format, C, As, bs; filefolder=output_folder)
        elseif sdp_format == "sdplr"
            write_problem_sdplr("G$i."*sdp_format, C, As, bs; filefolder=output_folder)
        else
            @error "Currently we only support sdpa/sdplr."
        end
    end
end



#"""
#Generate the SDPLR format for the maxcut SDP.
#"""
#function maxcut_write_sdplr(A::SparseMatrixCSC, filepath::String; Tv=Float64)
#    n = size(A, 1) 
#    d = sum(A, dims=2)[:, 1]
#    L = sparse(Diagonal(d) - A)
#    C = -Tv.(L)
#    open(filepath, "w") do f
#        write(f, "$n\n") # number of constraint matrices
#        write(f, "1\n")  # number of blocks in the SDP
#        write(f, "$n\n") # sizes of the blocks
#        write(f, ("1.0 "^n)*"\n") # b
#        write(f, "1.0\n")
#        triuC = triu(C)
#        write(f, "0 1 s $(nnz(triuC))\n")
#        for (i, j, v) in zip(findnz(triuC)...)
#            write(f, "$i $j $v\n")
#        end
#        for i = 1:n
#            write(f, "$i 1 s 1\n")
#            write(f, "$i $i 1.0\n")
#        end
#    end
#end