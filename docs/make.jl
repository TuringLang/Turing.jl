using Documenter, Turing

docpath = joinpath(@__DIR__, "doc")

makedocs(
    format = :html,
    sitename = "Turing.jl",
    pages = [
        "Home" => ["get-started.md",
                   "advanced.md",
                   "contributing/guide.md",
                   "contributing/style_guide.md",],
        "Tutorials" => ["ex/0_Introduction.md"],
        "API" => "api.md"
    ],
    build = "doc"
)

# Unused at current (2018-09-11) as our documentation serving solution doesn't
# play well with the deploydocs function.
# deploydocs(
#     repo = "github.com/cpfiffer/Turing.jl.git",
#     target = "build",
#     deps   = nothing,
#     make   = nothing,
#     julia = "1.0"
# )
