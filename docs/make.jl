using Documenter, Turing
makedocs(
    format = :html,
    sitename = "Turing.jl",
    pages = [
        "Home" => ["index.md",
                   "get-started.md",
                   "advanced.md",
                   "contributing/guide.md",
                   "contributing/style_guide.md",],
        "Tutorials" => ["ex/0_Introduction.md"],
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/cpfiffer/Turing.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing,
    julia = "1.0"
)
