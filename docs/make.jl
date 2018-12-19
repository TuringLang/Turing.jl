using Documenter, DocumenterMarkdown, DynamicHMC, Turing
using LibGit2: clone

# Include the update_homepage function.
include("make-utils.jl")

# Make paths.
examples_path = joinpath(@__DIR__, joinpath("site", "_tutorials"))
source_path = joinpath(@__DIR__, "src")
build_relative = "build"
build_path = joinpath(@__DIR__, build_relative)
site_path = joinpath(@__DIR__, "site")

# You can skip this part if you are on a metered
# connection by calling `julia make.jl no-tutorials`
if !in("no-tutorials", ARGS)
    isdir(examples_path) || mkpath(examples_path)

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
else
    @info "Skipping tutorial copying."
end

# Preprocess markdown files.
src_temp = mktempdir()
cp(source_path, src_temp, force = true)
yaml_dict = preprocess_markdown(source_path)

# Build documentation
try
    makedocs(
        sitename = "Turing.jl",
        build = build_relative,
        format = Markdown(),
        checkdocs = :all
    )
catch e
    # Put back the original files in the event of an error.
    cp(src_temp, source_path, force = true)
    rm(src_temp, recursive = true)
    rethrow(e)
end

# Postprocess markdown files (put the YAML headers back in)
cp(src_temp, source_path, force = true)
rm(src_temp, recursive = true)
postprocess_markdown(build_path, yaml_dict, original = source_path)

@info "YAML DICT"
for i in keys(yaml_dict)
    @info i
end

# Copy the built files to the site directory.
filecopy_deep(build_path, site_path)

if !in(ARGS, "no-publish")
    # Define homepage update function.
    page_update = update_homepage(
        "github.com/TuringLang/Turing.jl.git",
        "gh-pages",
        "site"
    )
else
    @info "Skipping publishing."
end
