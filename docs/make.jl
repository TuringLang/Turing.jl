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
        "API" => "functions.md"
    ]
)

deploydocs(
    repo = "github.com/cpfiffer/Turing.jl.git",
    julia = "1.0"
)
