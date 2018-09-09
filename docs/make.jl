using Documenter, Turing
makedocs(
<<<<<<< HEAD
    # format = :html,
=======
    format = :html,
>>>>>>> 27347ae4916a87ff9205bf45655e0028b3641f94
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
<<<<<<< HEAD

deploydocs(
    repo = "github.com/cpfiffer/Turing.jl.git",
    julia = "1.0"
)
=======
>>>>>>> 27347ae4916a87ff9205bf45655e0028b3641f94
