using Logging
using JSON
using GitHub

log_level =  Logging.Debug # Logging.Info | Logging.Warn | Logging.Error
global_logger(SimpleLogger(stdout, log_level))

# setup global variables
EVENT_FILE = get(ENV, "GITHUB_EVENT_PATH", "")
if EVENT_FILE == ""
    @error "No github event file specified."
    exit(0)
end

EVENT_DATA = JSON.Parser.parsefile(EVENT_FILE)
@debug EVENT_DATA

# Utilities

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

comment_url = EVENT_DATA["comment"]["html_url"]
comment_id = EVENT_DATA["comment"]["id"]
comment_body = EVENT_DATA["comment"]["body"]
user = EVENT_DATA["comment"]["user"]["login"]

reg_bm_cmd = r"!benchmark\((.*)\)"i
bm_cmd = match(reg_bm_cmd, comment_body)
if (bm_cmd != nothing)
    @info "benchmark command:", bm_cmd.captures[1]
    body = "Got benchmark command: $(bm_cmd.captures[1])"
    reply_comment(EVENT_DATA, body)
end


# Check if we should run benchmarks on current event


# Comment the benchmark results
# TODO
