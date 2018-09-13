using Documenter, Turing
using LibGit2: clone
using Weave

# Get path of documentation.
examples_path = joinpath(@__DIR__, joinpath("src", "ex"))

# Clone TuringTurorials
tmp_path = tempname()
mkdir(tmp_path)
clone("https://github.com/TuringLang/TuringTutorials", tmp_path)

# Weave all examples
try
    for file in readdir(tmp_path)
        if endswith(file, "ipynb")
            full_path = joinpath(tmp_path, file)
            Weave.weave(full_path,
                doctype = "hugo",
                out_path = examples_path,
                mod = Main)
        end
    end
catch e
    println("Weaving error: $e")
finally
    rm(tmp_path, recursive = true)
end

# Build documentation
makedocs(
    format = :html,
    sitename = "Turing.jl",
    pages = [
        "Home" => ["index.md",
                   "get-started.md",
                   "advanced.md",
                   "contributing/guide.md",
                   "contributing/style_guide.md",],
        "Tutorials" => ["ex/tutorials.md",
                        "ex/0_Introduction.md"],
        "API" => "api.md"
    ],
    build = "doc"
)
