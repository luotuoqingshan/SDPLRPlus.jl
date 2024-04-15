using Plots

x = 1:0.1:10;
y = rand(length(x));

plot(x, y, xscale=:log2, xticks=(1:10, 1:10))

function myadd(A::SparseMatrixCSC{Tv, Ti}, B::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    ...
end