# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

# If your source directory is not accessible through 
# Julia's LOAD_PATH, you might wish to add the following line 
# at the top of make.jl
push!(LOAD_PATH,"../src/")

using Documenter
using DocumenterCitations
using SDPLRPlus


bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:authoryear
)
makedocs(
    modules = [SDPLRPlus],
    authors = "Yufan Huang and the SDPLR authors",
    sitename = "SDPLRPlus.jl",
    format=Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical="https://luotuoqingshan.github.io/SDPLRPlus.jl",
        assets=String[],
    ),
    pages = ["index.md"];
    plugins=[bib],
    checkdocs=:exports,
)

# Some setup is needed for documentation deployment, see “Hosting Documentation” and
# deploydocs() in the Documenter manual for more information.
deploydocs(;
    repo = "github.com/luotuoqingshan/SDPLRPlus.jl",
    devbranch = "main",
)
