module SDPLR
    using Random                      

    # linear algebra
    using SparseArrays, LinearAlgebra, LuxurySparse, MKLSparse, MKL

    # eig computation
    using GenericArpack

    # key data structures defined for the solver 
    include("structs.jl")
    export SymLowRankMatrix

    # core operations
    include("coreop.jl")

    # L-BFGS
    include("lbfgs.jl")

    # line search
    using Polynomials, PolynomialRoots
    include("linesearch.jl")

    # printing functions
    using Printf
    include("myprint.jl")

    # options
    using Parameters
    include("options.jl")

    # preprocessing
    include("preprocess.jl")

    # utils 
    include("utils.jl")

    # main function
    include("sdplr.jl")

    export sdplr
end
