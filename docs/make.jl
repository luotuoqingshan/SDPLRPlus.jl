# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

# If your source directory is not accessible through 
# Julia's LOAD_PATH, you might wish to add the following line 
# at the top of make.jl
push!(LOAD_PATH,"../src/")

using Documenter
using DocumenterCitations
using LuxurySparse 
using SDPLRPlus


bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))
makedocs(
    modules = [SDPLRPlus, LuxurySparse],
    authors = "Yufan Huang and the SDPLR authors",
    sitename = "SDPLRPlus.jl",
    format=Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical="https://luotuoqingshan.github.io/SDPLRPlus.jl",
        assets=String[],
    ),
    plugins=[bib],
    pages = ["index.md"]
)

# Some setup is needed for documentation deployment, see “Hosting Documentation” and
# deploydocs() in the Documenter manual for more information.
deploydocs(;
    repo = "github.com/luotuoqingshan/SDPLRPlus.jl",
    devbranch = "main",
)
