using Dates
using Logging
using JSON

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

# Extract event information

if EVENT_DATA["action"] != "created" || !haskey(EVENT_DATA, "comment")
    @debug "Not an event for comment creation"
    exit(0)
end

user = EVENT_DATA["comment"]["user"]["login"]
tags, branches = BenchmarkHelper.benchmarks_info(EVENT_DATA)
job_id =  Dates.format(Dates.now(), "YmmddHHMMSS")

if (branches != nothing && !isempty(branches))
    @info "benchmark target branches:", branches
    body = [
        "Hi @$user, I just got a benchmark command from you,"
        "I will run a benchmark(JOB_ID=$(job_id)) on branches $branches as soon as possible."
    ]
    BenchmarkHelper.reply_comment(EVENT_DATA, join(body, "\n"))
else
    @info "No benchmarking command found in the comment."
    body = [
        "Hi @$user, I just got a benchmark command from you,"
        "But I can't determine on which branches to run,"
        "Please specify the branches in your command."
    ]
    BenchmarkHelper.reply_comment(EVENT_DATA, join(body, "\n"))
    exit(0)
end

# Run benchmarks on current event
BenchmarkHelper.set_benchmark_info("", job_id)
BenchmarkHelper.run_benchmarks(tags, branches)

# Comment the benchmark results
BenchmarkHelper.reply_comment(EVENT_DATA, BenchmarkHelper.get_results(job_id, user))
