# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, SDPLR-jl

makedocs(
    modules = [SDPLR-jl],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "luotuoqingshan",
    sitename = "SDPLR-jl.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

# Some setup is needed for documentation deployment, see “Hosting Documentation” and
# deploydocs() in the Documenter manual for more information.
deploydocs(
    repo = "github.com/luotuoqingshan/SDPLR-jl.jl.git",
    push_preview = true
)
