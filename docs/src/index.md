# Documentation

## Main interface
```@docs
sdplr
```

## Supported constraint types

As mentioned, `SDPLRPlus` supports semidefinite programs with constraints of type
- [`Diagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Diagonal), 
- [`SparseMatrixCSC`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/#man-csc), 
- [`SparseMatrixCOO`](https://yaoquantum.org/LuxurySparse.jl/latest/luxurysparse/#LuxurySparse.SparseMatrixCOO) and 
- `SymLowRankMatrix`. 
`SymLowRankMatrix` is a type for symmetric low-rank matrices defined in `SDPLRPlus`.
```@docs
SymLowRankMatrix
```

## Citing `SDPLRPlus` 
`SDPLRPlus` is introduced in our paper [huang2024suboptimality](@cite). If you found it useful in your work, we kindly request you to cite it.

Moreover, `SDPLRPlus` started as a reproduction of the package 
`SDPLR` which was introduced in [BMNonlinear2003, BMLocal2005, BCComputational2006](@cite). Please consider citing these papers. 

## References
```@bibliography
```