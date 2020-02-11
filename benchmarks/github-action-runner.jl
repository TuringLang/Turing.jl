using Logging
using JSON
using GitHub

push!(LOAD_PATH, abspath(@__DIR__))
using BenchmarkHelper

log_level =  Logging.Debug # Logging.Info | Logging.Warn | Logging.Error
global_logger(SimpleLogger(stdout, log_level))

# setup global variables
EVENT_FILE = get(ENV, "GITHUB_EVENT_PATH", "")
if isempty(EVENT_FILE)
    @error "No github event file specified."
    exit(0)
end

EVENT_DATA = JSON.Parser.parsefile(EVENT_FILE)
@debug EVENT_DATA

# Utilities

function bm_branches(br_text)
    branches = filter(!isempty, map(strip, split(br_text, ",")))
    if isempty(branches)
        return ["HEAD", "master"]
    elseif length(branches) == 1
        if "master" in branches
            return push!(branches, "master")
        end
        return push!(branches, "HEAD")
    else
        return branches
    end
end

function bm_benchmarks(bm_text)
    return filter(!isempty, map(strip, split(bm_text, ",")))
end

function reply_comment(event_data, body)
    repo = event_data["repository"]["full_name"]
    auth = GitHub.authenticate(get(ENV, "GITHUB_TOKEN", ""))
    params = Dict("body" => body)

    comment_kind = :issue
    comment_parent = nothing

    if haskey(EVENT_DATA, "issue")
        comment_kind = :issue
        comment_parent = event_data["issue"]["number"]
    elseif haskey(event_data["comment"], "commit_id")
        comment_kind = :commit
        comment_parent = event_data["comment"]["commit_id"]
    end

    create_comment(repo, comment_parent, comment_kind; params=params, auth=auth)
end

# Extract event information

if EVENT_DATA["action"] != "created" || !haskey(EVENT_DATA, "comment")
    @debug "Not an event for comment creation"
    exit(0)
end

comment_body = EVENT_DATA["comment"]["body"]
user = EVENT_DATA["comment"]["user"]["login"]

reg_bm_cmd = r"!benchmark\((.*)\)"i
bm_cmd = match(reg_bm_cmd, comment_body)
if (bm_cmd != nothing)
    @info "benchmark command:", bm_cmd.match
    branches = bm_branches(bm_cmd.captures[1])
    body = [
        "Hi @$user, I just got a benchmark command from you: `$(bm_cmd.match)`"
        "I will run a benchmark on branches $branches as soon as possible."
    ]
    reply_comment(EVENT_DATA, join(body, "\n"))
else
    @info "No benchmarking command found in the comment."
    exit(0)
end

# Run benchmarks on current event
result_files = []
# TODO
BenchmarkHelper.run_benchmark("benchmarks/dummy.jl")

# Comment the benchmark results
reply_comment(EVENT_DATA, BenchmarkHelper.get_results(user))
