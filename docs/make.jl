using Documenter, Turing
makedocs(
    format = :html,
    sitename = "Turing.jl",
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["ex/introduction.md",
        ]
    ]
)
