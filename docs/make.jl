using Documenter
using Turing

using DocumenterInterLinks

links = InterLinks(
    "DynamicPPL" => "https://turinglang.org/DynamicPPL.jl/stable/",
    "AbstractPPL" => "https://turinglang.org/AbstractPPL.jl/stable/",
    "LinearAlgebra" => "https://docs.julialang.org/en/v1/",
    "AbstractMCMC" => "https://turinglang.org/AbstractMCMC.jl/stable/",
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/",
    "AdvancedVI" => "https://turinglang.org/AdvancedVI.jl/stable/",
    "FlexiChains" => "https://pysm.dev/FlexiChains.jl/stable/",
    "DistributionsAD" => "https://turinglang.org/DistributionsAD.jl/stable/",
    "OrderedCollections" => "https://juliacollections.github.io/OrderedCollections.jl/stable/",
    "Distributions" => "https://juliastats.org/Distributions.jl/stable/",
)

# Doctest setup
DocMeta.setdocmeta!(Turing, :DocTestSetup, :(using Turing); recursive=true)

makedocs(;
    sitename="Turing",
    modules=[Turing],
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Submodule APIs" => [
            "Inference" => "api/Inference.md",
            "Optimisation" => "api/Optimisation.md",
            "Variational " => "api/Variational.md",
            "RandomMeasures " => "api/RandomMeasures.md",
        ],
    ],
    checkdocs=:exports,
    doctest=false,
    warnonly=true,
    plugins=[links],
)
