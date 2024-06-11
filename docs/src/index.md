# Documentation

## Main interface
```@docs
sdplr
```

## Supported constraint types

As mentioned, `SDPLRPlus` supports semidefinite programs with constraints of type
- `Diagonal`, 
- `SparseMatrixCSC`, 
- `SparseMatrixCOO` and 
- `SymLowRankMatrix`. 
`SymLowRankMatrix` is a type for symmetric low-rank matrices defined in `SDPLRPlus`.
```@docs
SymLowRankMatrix
```
Because `SparseMatrixCOO` is less commonly used, here we include its docstring for completeness.
```@docs
SparseMatrixCOO
```

## Citing `SDPLRPlus` 