using Documenter
using Turing

using DocumenterInterLinks

links = InterLinks(
    "DynamicPPL" => "https://turinglang.org/DynamicPPL.jl/stable/objects.inv",
    "AbstractPPL" => "https://turinglang.org/AbstractPPL.jl/stable/objects.inv",
    "LinearAlgebra" => "https://docs.julialang.org/en/v1/objects.inv",
    "AbstractMCMC" => "https://turinglang.org/AbstractMCMC.jl/stable/objects.inv",
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/objects.inv",
    "AdvancedVI" => "https://turinglang.org/AdvancedVI.jl/stable/objects.inv",
    "DistributionsAD" => "https://turinglang.org/DistributionsAD.jl/stable/objects.inv",
    "OrderedCollections" => "https://juliacollections.github.io/OrderedCollections.jl/stable/objects.inv",
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
        ],
    ],
    checkdocs=:exports,
    doctest=false,
    warnonly=true,
    plugins=[links],
)
