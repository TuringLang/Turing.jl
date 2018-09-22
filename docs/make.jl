using Documenter, Turing
using LibGit2: clone

# Get paths.
examples_path = joinpath(@__DIR__, joinpath("src", "ex"))

# Clone TuringTurorials
tmp_path = tempname()
mkdir(tmp_path)
clone("https://github.com/TuringLang/TuringTutorials", tmp_path)

# Move to markdown folder.
md_path = joinpath(tmp_path, "markdown")

# Copy the .md versions of all examples.
try
    println(md_path)
    for file in readdir(md_path)
        full_path = joinpath(md_path, file)
        target_path = joinpath(examples_path, file)
        println("Copying $full_path to $target_path")
        cp(full_path, target_path, force = true)
    end
catch e
    # println("Markdown copy error: $e")
    rethrow(e)
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
                   "guide.md",
                   "advanced.md",
                   "contributing/guide.md",
                   "contributing/style_guide.md",],
        "Tutorials" => ["ex/0_Introduction.md"],
        "API" => "api.md"
    ]
)

# Deploy documentation.
deploydocs(
    repo   = "github.com/TuringLang/Turing.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing,
    julia  = "1.0"
)
