using GZip, ZipFile, CSV, MAT, DataFrames
using SparseArrays, LinearAlgebra, MatrixNetworks

function read_txt_gz(
    filename::String;
    filefolder::String=homedir()*"/datasets/raw/",
)
    filepath = filefolder*filename*".txt.gz"
    linecounter = 0
    I = Int64[]    
    J = Int64[]
    GZip.open(filepath) do f
        for l in eachline(f)
            linecounter += 1
            if linecounter >= 5                 
                u, v = split(l, '\t')
                u = parse(Int, u)
                v = parse(Int, v)
                push!(I, u)
                push!(J, v)
            end
        end
    end
    I .+= 1
    J .+= 1
    n = max(maximum(I), maximum(J))
    A = sparse(I, J, ones(Float64, length(I)), n, n)
    return A
end


function read_ungraph_txt_gz(
    filename::String;
    filefolder::String=homedir()*"/datasets/raw/",
)
    filepath = filefolder*filename*".ungraph.txt.gz"
    linecounter = 0
    I = Int64[]    
    J = Int64[]
    GZip.open(filepath) do f
        for l in eachline(f)
            linecounter += 1
            if linecounter >= 5                 
                u, v = split(l, '\t')
                u = parse(Int, u)
                v = parse(Int, v)
                push!(I, u)
                push!(J, v)
                if u != v
                    push!(J, u)
                    push!(I, v)
                end
            end
        end
    end
    minid = min(minimum(I), minimum(J))-1
    shift = max(0, -minid)
    I .+= shift
    J .+= shift
    n = max(maximum(I), maximum(J))
    A = sparse(I, J, ones(Float64, length(I)), n, n)
    return A
end


function read_zip(
    filename::String;
    filefolder::String=homedir()*"/datasets/raw/",
)
    zip_filename = filefolder*filename*".zip"
    z = ZipFile.Reader(zip_filename)
    for f in z.files
        names = split(f.name, "/")
        if names[end] == filename*"_edges.csv"
            df = CSV.read(f, DataFrame)
            I = df[1:end, 1] 
            J = df[1:end, 2] 
            minid = min(minimum(I), minimum(J))-1
            shift = max(0, -minid)
            I .+= shift
            J .+= shift
            n = max(maximum(I), maximum(J))
            A = sparse(I, J, ones(Float64, length(I)), n, n)
            A = max.(A, A')
            return A
        end
    end
end


function read_gset(
    filename::String;
    filefolder::String=homedir()*"/Gset/",
)::SparseMatrixCSC
    I = Int32[] 
    J = Int32[]
    V = Float64[]
    n = 0
    filepath = filefolder*filename
    open(filepath) do file
        lines = readlines(file)
        n, _ = split(lines[1], ' ')
        n = parse(Int, n)
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


function postprocess_graph(A)
    B = deepcopy(A)

    # take the largest component
    B, _ = largest_component(B)
    n = size(B, 1)
    oldm = nnz(B) 
    @info "The largest component's n = $(n) \
        number of directed edges(non self-loop counts as 2) = $(oldm)"

    # remove self-loops
    B[diagind(B)] .= 0
    dropzeros!(B)
    newm = nnz(B)
    @info "Removed $(oldm - newm) self-loops. The number of edges = $(div(newm, 2))"
    return B 
end

for dataset in ["ca-AstroPh", "ca-CondMat", "ca-GrQc", "ca-HepPh", "ca-HepTh",
    "email-Enron"]
    @info dataset
    A = read_txt_gz(dataset)
    B = postprocess_graph(A)
    matwrite(homedir()*"/datasets/graphs/"*dataset*".mat", Dict("A" => B))
end

for dataset in ["com-amazon", "com-dblp", "com-lj", "com-orkut", "com-youtube"]
    @info dataset
    A = read_ungraph_txt_gz(dataset)
    B = postprocess_graph(A)
    matwrite(homedir()*"/datasets/graphs/"*dataset*".mat", Dict("A" => B))
end

for dataset in ["deezer_europe", "musae_facebook"]
    @info dataset
    A = read_zip(dataset)
    B = postprocess_graph(A)
    matwrite(homedir()*"/datasets/graphs/"*dataset*".mat", Dict("A" => B))
end


for dataset in ["web-BerkStan", "web-Google", "web-NotreDame", "web-Stanford"]
    @info dataset
    A = read_txt_gz(dataset)
    A = max.(A, A')
    B = postprocess_graph(A)
    matwrite(homedir()*"/datasets/graphs/"*dataset*".mat", Dict("A" => B))
end

for i in [(1:67)..., 70, 72, 77, 81]
    dataset = "G$i" 
    @info dataset
    A = read_gset(dataset)
    matwrite(homedir()*"/datasets/graphs/"*dataset*".mat", Dict("A" => A))
end

for (root, dirs, files) in walkdir(homedir()*"/datasets/graphs/")
    for file in files
        @info file
        data = matread(root*"/"*file)
        A = data["A"]
        A_abs = abs.(A)
        A_dummy = A
        n = size(A, 1)
        if mod(n, 2) == 1
            I, J, V = findnz(A)
            A_dummy = sparse(I, J, V, n+1, n+1)
        end
        A_dummy_abs = abs.(A_dummy)
        matwrite(root*"/"*file,
            Dict("A" => A, "A_abs" => A_abs, "A_dummy" => A_dummy, "A_dummy_abs" => A_dummy_abs)
        )
    end
end