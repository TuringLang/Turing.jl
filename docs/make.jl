using Documenter, Turing
using LibGit2: clone
using Weave

# Get path of documentation.
examples_path = joinpath(@__DIR__, joinpath("src", "ex"))

# Clone TuringTurorials
tmp_path = tempname()
mkdir(tmp_path)
clone("https://github.com/TuringLang/TuringTutorials", tmp_path)

function polish_latex(path::String)
    txt = open(f -> read(f, String), path)
    open(path, "w+") do f
        write(f, replace(txt, raw"$$" => raw"\$\$"))
    end
end

# Weave all examples
try
    for file in readdir(tmp_path)
        if endswith(file, "ipynb")
            out_name = split(file, ".")[1] * ".md"
            out_path = joinpath(examples_path, out_name)

            full_path = joinpath(tmp_path, file)

            Weave.weave(full_path,
                doctype = "hugo",
                out_path = out_path,
                mod = Main)

            polish_latex(out_path)
        end
    end
catch e
    println("Weaving error: $e")
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
                   "advanced.md",
                   "contributing/guide.md",
                   "contributing/style_guide.md",],
        "Tutorials" => ["ex/0_Introduction.md"],
        "API" => "api.md"
    ],
    build = "doc"
)
