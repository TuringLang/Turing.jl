module BenchmarkHelper

using Pkg
using GitHub

export PROJECT_DIR,
    log_report, save_results, @benchmark_expr,
    target_branches, reply_comment,
    benchmark_files, run_benchmark, run_benchmarks, get_results

const PROJECT_DIR = abspath(@__DIR__) |> dirname
const CURRENT_BRANCH = Ref("HEAD")
const CURRENT_BM = Ref("")
const REPORTED_LOGS = String[]

CONFIG = Pkg.TOML.parsefile(joinpath(PROJECT_DIR, "benchmarks/config.toml"))

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

    data["benchmark_file"] = CURRENT_BM[]
    data["branch"] = CURRENT_BRANCH[]
    @info "benchmark result" data
    open(output_path("result"), "w") do io
        Pkg.TOML.print(io, data)
    end
end

macro benchmark_expr(engine, expr)
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

function target_branches(event_data)
    comment_body = event_data["comment"]["body"]
    reg_bm_cmd = r"!benchmark\((.*)\)"i
    bm_cmd = match(reg_bm_cmd, comment_body)

    if (bm_cmd != nothing)
        @info "benchmark command:", bm_cmd.match
    else
        return nothing
    end

    branches = filter(!isempty, map(x -> strip(x, [' ', '"']),
                                    split(bm_cmd.captures[1], ",")))
    target = "master"
    if haskey(event_data, "pull_request")
        target = event_data["pull_request"]["head"]["ref"]
    end
    if isempty(branches)
        if target != "master"
            return [target, "master"]
        end
    elseif length(branches) == 1
        if !in(target, branches)
            return push!(branches, target)
        end
        if !in("master", branches)
            return push!(branches, "master")
        end
    else
        return branches
    end
end

function target_benchmarks(bm_text)
    return filter(!isempty, map(strip, split(bm_text, ",")))
end

function reply_comment(event_data, body)
    repo = event_data["repository"]["full_name"]
    auth = GitHub.authenticate(get(ENV, "GITHUB_TOKEN", ""))
    params = Dict("body" => body)

    comment_kind = :issue
    comment_parent = nothing

    if haskey(event_data, "issue")
        comment_kind = :issue
        comment_parent = event_data["issue"]["number"]
    elseif haskey(event_data["comment"], "commit_id")
        comment_kind = :commit
        comment_parent = event_data["comment"]["commit_id"]
    end

    create_comment(repo, comment_parent, comment_kind;
                   params=params, auth=auth)
end

function benchmark_excluded(file)
    excludes = joinpath(PROJECT_DIR, "benchmarks", "excludes.txt")
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

function run_benchmark(file)
    empty!(REPORTED_LOGS)
    code = """
    using Pkg;
    pkg"instantiate; add JSON GitHub BenchmarkTools;"
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

function run_benchmarks(branches)
    for branch in branches
        run(`git checkout .`)
        run(`git checkout $branch`)
        for bm in benchmark_files()
            run_benchmark(joinpath("benchmarks", bm))
        end
    end
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
