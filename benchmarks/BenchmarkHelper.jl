module BenchmarkHelper

using Pkg

export PROJECT_DIR,
    log_report, save_results, @tbenchmark_expr,
    benchmark_files, run_benchmark, get_results

const PROJECT_DIR = abspath(@__DIR__) |> dirname
const CURRENT_BRANCH = Ref("HEAD")
const CURRENT_BM = Ref("")
const REPORTED_LOGS = String[]

NON_BENCHMARK_FILES = [
    "BenchmarkHelper.jl",
    "github-action-runner.jl",
    "runbenchmarks.jl",
]

# Functions run in benchmark processes

function set_benchmark_info(branch, file)
    CURRENT_BRANCH[] = branch
    CURRENT_BM[] = file
end

function output_path(suffix="unknown")
    branch = replace(CURRENT_BRANCH[], r"[-/.]"=>"_")
    filename = replace(CURRENT_BM[], r"[-/.]"=>"_")
    outdir= joinpath(PROJECT_DIR, "benchmarks", "output")
    mkpath(outdir)
    return joinpath(outdir, "$branch-$filename-$suffix.txt")
end

function log_report(log::String)
    @info "log reported" log
    push!(REPORTED_LOGS, log)
end

function save_results(data::Dict)
    open(output_path("log"), "w") do io
        map(REPORTED_LOGS) do line
            write(io, line)
            write(io, "\n")
        end
    end
    @info "benchmark result" data
    open(output_path("result"), "w") do io
        Pkg.TOML.print(io, data)
    end
end

macro tbenchmark_expr(engine, expr)
    quote
        chain, t_elapsed, mem, gctime, memallocs  = @timed $(esc(expr))
        Dict(
            "engine" => $(string(engine)),
            "time" => t_elapsed,
            "memory" => mem,
        )
    end
end


# Functions run in the dispather process

function gitcurrentbranch()
    branches = readlines(`git branch`)
    ind = findfirst(x -> x[1] == '*', branches)
    brname = branches[ind][3:end]
    if brname[1] == '('
        brname = readlines(`git rev-parse --short HEAD`)[1]
    end
    return brname
end

function benchmark_files()
    filter(readdir(joinpath(PROJECT_DIR, "benchmarks"))) do file
        !in(file, NON_BENCHMARK_FILES) && endswith(file, ".jl")
    end
end

function run_benchmark(file)
    empty!(REPORTED_LOGS)
    code = """
    using Pkg;
    pkg"instantiate; add JSON GitHub;"
    Pkg.build(verbose=true)
    push!(LOAD_PATH, joinpath("$(PROJECT_DIR)", "benchmarks"))
    using BenchmarkHelper
    BenchmarkHelper.set_benchmark_info("$(gitcurrentbranch())", "$file")
    include("$(file)")
    save_results(BENCHMARK_RESULT)
    """
    job = `julia --project=$(PROJECT_DIR) -e $code`
    @info(job);
    run(job)
    @info("run `$file` ✓")
end

function get_results(user)
    body = """
### Benchmark Results

Hi @$user, here are the results of your demanded benchmarks:
"""

    outdir= joinpath(PROJECT_DIR, "benchmarks", "output")
    for file in readdir(outdir)
        body *= "\nContent of file `$file`:\n"
        body *= "\n```\n"
        body *= read(joinpath(outdir, file), String)
        body *= "\n```\n"
    end
    return body
end

end
