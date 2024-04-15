include("header.jl")
using Scanf

function read_SDPLR_res(filename::String)
    m = 801
    n = 800
    r = 10
    y = zeros(Float64, m)
    R = zeros(Float64, n, r)
    open(filename, "r") do f
        lines = readlines(f)
        for i = 1:m
            y[i] = parse(Float64, lines[i+1]) 
        end
        for j = 1:r
            for i = 1:n
                R[i, j] = parse(Float64, lines[(j-1)*n+i+m+2])
            end
        end
    end
    return R, y
end 

R, y = read_SDPLR_res(homedir()*"/SDPLR-1.03-beta/G1_minbisec.sol")

A = read_graph("G1")
C, As, bs = minimum_bisection(A)

res, SDP = sdplr(C, As, bs, 10)
SDP.λ .= y
SDP.R .= R

normC = norm(C, 2)
normb = norm(bs, 2)

fg!(SDP, normC, normb)

surrogate_duality_gap(SDP, 800.0, 2000;highprecision=true)