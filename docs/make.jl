using Documenter
using Turing
# Need to import these packages to generate docs for functions
# from those packages.
using Bijectors
using Distributions
using DynamicPPL

using DocumenterInterLinks

links = InterLinks(
    "DynamicPPL" => "https://turinglang.org/DynamicPPL.jl/stable/objects.inv",
    "AbstractPPL" => "https://turinglang.org/AbstractPPL.jl/dev/objects.inv",
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/objects.inv",
    "AdvancedVI" => "https://turinglang.org/AdvancedVI.jl/v0.2.8/objects.inv",
    "DistributionsAD" => "https://turinglang.org/DistributionsAD.jl/stable/objects.inv",
)

# Doctest setup
DocMeta.setdocmeta!(Turing, :DocTestSetup, :(using Turing, DynamicPPL); recursive=true)

makedocs(;
    sitename="Turing",
    modules=[Turing, Distributions, Bijectors],
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Submodule APIs" =>
            ["Inference" => "api/Inference.md", "Optimisation" => "api/Optimisation.md"],
    ],
    checkdocs=:exports,
    # checkdocs_ignored_modules=[Turing, Distributions, DynamicPPL, AbstractPPL, Bijectors],
    doctest=false,
    warnonly=true,
    plugins=[links],
)
