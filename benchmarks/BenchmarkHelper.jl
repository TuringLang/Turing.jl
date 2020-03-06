module BenchmarkHelper

using Pkg
using BenchmarkTools
using GitHub

export PROJECT_DIR, BenchmarkSuite, benchmark_files


const PROJECT_DIR = abspath(@__DIR__) |> dirname
const CONFIG = Pkg.TOML.parsefile(joinpath(PROJECT_DIR, "benchmarks/config.toml"))

# Global Benchmark Suite
const BenchmarkSuite = BenchmarkGroup()

# Functions run in the dispather process

function benchmark_excluded(file)
    patterns = get(CONFIG, "EXCLUDES", [])
    for pat in patterns
        match(Regex(pat), file) != nothing && return true
    end
    return false
end

function benchmark_files()
    bm_files = []
    for (root, dirs, files) in walkdir(joinpath(PROJECT_DIR, "benchmarks"))
        for file in files
            push!(bm_files,joinpath(root, file))
        end
    end

    bm_files = map(bm_files) do path
        replace(path, joinpath(PROJECT_DIR, "benchmarks/") => "")
    end

    filter(bm_files) do file
        endswith(file, ".jl") && !benchmark_excluded(file)
    end
end

end
