module BenchmarkHelper

using BenchmarkTools


using Pkg
using GitHub

export PROJECT_DIR, BenchmarkSuite,
    benchmarks_info, reply_comment,
    benchmark_files, run_benchmarks, get_results


include("ns-utilities.jl")

const PROJECT_DIR = abspath(@__DIR__) |> dirname
const CURRENT_BRANCH = Ref("HEAD")
const CURRENT_JOB_ID = Ref("NEW-ID")
const CONFIG = Pkg.TOML.parsefile(joinpath(PROJECT_DIR, "benchmarks/config.toml"))

# Global Benchmark Suite
const BenchmarkSuite = BenchmarkGroup()

# Functions run in benchmark processes

function set_benchmark_info(branch, job_id)
    CURRENT_BRANCH[] = branch
    CURRENT_JOB_ID[] = job_id
end

function output_path()
    branch = replace(CURRENT_BRANCH[], r"[-/.]"=>"_")
    outdir= joinpath(PROJECT_DIR, "benchmarks", "output")
    mkpath(outdir)
    return joinpath(outdir, "BM-$(CURRENT_JOB_ID[])-$branch.txt")
end

function run_suite(tags_expr=:ALL)
    benchmarks = eval(Meta.parse("BenchmarkSuite[@tagged $tags_expr]"))
    t = run(benchmarks, samples=1)
    branch = CURRENT_BRANCH[]
    @info "benchmark result", t
    open(output_path(), "w") do io
        show(io, bmresults_to_dataframe(t))
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

function command_info(event_data)
    comment_body = event_data["comment"]["body"]
    reg_bm_cmd = r"`\w+\(.*\)`"i
    bm_cmd = match(reg_bm_cmd, comment_body)
    bm_cmd == nothing && return (nothing, nothing, nothing)
    cmd, args, kwargs = parse_submission_string(bm_cmd.match)
    return (cmd, args, kwargs)
end

function benchmarks_info(event_data)
    cmd, args, kwargs = command_info(event_data)
    if cmd != "runbenchmarks"
        return false, nothing, nothing
    end

    branches =  haskey(kwargs, :vs) ? Meta.parse(kwargs[:vs]) : []
    if branches isa Expr && branches.head == :vect
        branches = branches.args
    elseif branches isa AbstractString
        branches = [branches]
    end
    target = "master"
    if haskey(event_data, "pull_request")
        target = event_data["pull_request"]["head"]["ref"]
    end
    if isempty(branches)
        if target != "master"
            branches = [target, "master"]
        end
    elseif length(branches) == 1
        if !in(target, branches)
            push!(branches, target)
        elseif !in("master", branches)
            push!(branches, "master")
        end
    end
    return true, Meta.parse(args[1]), branches
end

function target_benchmarks(bm_text)
    return filter(!isempty, map(strip, split(bm_text, ",")))
end

function reply_comment(event_data, body)
    repo = event_data["repository"]["full_name"]
    # auth = GitHub.authenticate(get(ENV, "GITHUB_TOKEN", ""))
    auth = GitHub.authenticate(get(ENV, "BENCHMARK_KEY", ""))
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

function run_benchmarks(tags, branches)
    job_id = CURRENT_JOB_ID[]
    for branch in branches
        branch = replace(branch, r"^.*?[:@#]"i => "")
        run(`git checkout .`)
        run(`git checkout $branch`)
        includes = ""
        for bm in benchmark_files()
            includes *= """include("$(joinpath("benchmarks", bm))")\n"""
        end

        code = """
        using Pkg;
        pkg"instantiate; add JSON GitHub DataFrames BenchmarkTools;"
        Pkg.build(verbose=true)
        push!(LOAD_PATH, joinpath("$(PROJECT_DIR)", "benchmarks"))
        using BenchmarkHelper
        BenchmarkHelper.set_benchmark_info("$(gitcurrentbranch())", "$(job_id)")
        $(includes)
        BenchmarkHelper.run_suite(:($tags))
        """
        job = `julia --project=$(PROJECT_DIR) -e $code`
        @info(job);
        run(job)
    end
end

function get_results(job_id, user)
    # TODO
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
