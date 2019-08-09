using Documenter, DocumenterMarkdown, DynamicHMC, Turing
using LibGit2: clone

# Include the utility functions.
include("make-utils.jl")

# Paths.
source_path = joinpath(@__DIR__, "src")
build_path = joinpath(@__DIR__, "_docs")
tutorial_path = joinpath(@__DIR__, "_tutorials")

with_clean_docs(source_path, build_path) do source, build
    makedocs(
        sitename = "Turing.jl",
        source = source,
        build = build,
        format = Markdown(),
        checkdocs = :all
    )
end

# You can skip this part if you are on a metered
# connection by calling `julia make.jl no-tutorials`
in("no-tutorials", ARGS) || copy_tutorial(tutorial_path)

# Copy the built files to the site directory.
# filecopy_deep(build_path, site_path)

if false && !in(ARGS, "no-publish")
    # Define homepage update function.
    page_update = update_homepage(
        "github.com/TuringLang/Turing.jl.git",
        "gh-pages",
        "site"
    )
else
    @info "Skipping publishing."
end
