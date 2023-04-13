include("structures.jl")


"""
L-BFGS two-loop recursion
to compute the direction

q = nabla f_k
for i = k - 1, k - 2, ... k - m
    alpha_i = rho_i * s_i^T q 
    q = q - alpha_i * y_i
end
r = H_k * q # we don't do this step

for i = k - m, k - m + 1, ... k - 1
    beta_i = rho_i * y_i^T r
    r = r + s_i * (alpha_i - beta_i)
end

Notice here if we initialize all s_i, y_i to be zero
then we don't need to record how many
s_i, y_i pairs we have already computed 
"""
function dirlbfgs(
    algdata::AlgorithmData,
    lbfgshis::lbfgshistory;
    negate::Int=1,
)
    # we store l-bfgs vectors as a cyclic array
    dir = algdata.G
    m = lbfgshis.m
    lst = lbfgshis.latest
    # pay attention here, dir, s and y are all matrices
    for i = 1:m
        j = mod(lst - i + m, m) + 1 # j = (lst - (i-1)) 
        alpha = lbfgshis.vecs[j].rho * (lbfgshis.vecs[j].s .* dir)
        dir -= lbfgshis.vecs[j].y * alpha 
        lbfghis.vecs[j].a = alpha
    end

    for i = m:1
        j = mod(lst - i + m, m) + 1 
        beta = lbfgshis.vecs[j].rho * (lbfgshis.vecs[j].y .* dir)
        dir += lbfgshis.vecs[j].s * (lbfghis.vec[i].a - beta) 
    end

    # we need to pick -dir as search direction
    if negate 
        dir .*= -1
    end
    j = (lbfgshis.latest) % lbfgshis.m + 1
    lbfgshis.vecs[j].y = -algdata.G
    return dir
end


function lbfgs_postprocess!(
    algdata::AlgorithmData,
    lbfgshis::lbfgshistory,
    dir::Matrix{Float64},
    stepsize::Float64,
)
    # update lbfgs history
    j = (lbfgshis.latest) % lbfgshis.m + 1
    lbfgshis.vecs[j].s = stepsize * dir
    lbfgshis.vecs[j].y += algdata.G
    lbfgshis.vecs[j].rho = 1 / (lbfgshis.vecs[j].y' * lbfgshis.vecs[j].s)
    lbfgshis.latest = j
end

