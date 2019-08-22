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

baseurl = "/dev"
if get(ENV, "TRAVIS_TAG", "") != ""
    baseurl = ENV["TRAVIS_TAG"]
end
jekyll_build = joinpath(@__DIR__, "jekyll-build")
with_baseurl(() -> run(`$jekyll_build`), baseurl)

# deploy
devurl = "dev"
repo = "github.com:TuringLang/Turing.jl.git"
deploydocs(
    target = "_site",
    repo = repo,
    branch = "gh-pages",
    devbranch = "master",
    devurl = devurl,
    versions = ["stable" => "v^", "v#.#", devurl => devurl]
)
