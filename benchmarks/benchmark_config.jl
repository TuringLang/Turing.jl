ON_TRAVIS = get(ENV, "TRAVIS", "false") == "true"

if ON_TRAVIS
    BENCHMARK_FILES = [
        "dummy.jl",
        "gdemo.jl",
        "mvnormal.jl",
    ]
else
    BENCHMARK_FILES = [
        "dummy.jl",
        "gdemo.jl",
        "mvnormal.jl",
    ]
end

BENCHMARK_FILES = map(BENCHMARK_FILES) do x joinpath(@__DIR__, x) end


COMMENT_TEMPLATES = Dict{Symbol, String}()
COMMENT_TEMPLATES[:CI_BM_COMPLETE_FOR_COMMIT] = """
!!!THIS IS A CUSTOMIZED COMMENT IN "$(@__FILE__)"!!!

The benchmark job for this commit is finished.

The report is committed in this commit: {{{ :report_repo }}}#{{{ :commit_id }}}.

You can see the report at {{{ :report_url }}}.

Below is the log reported from the benchmark:

```
{{#:report_logs}}
{{.}}
{{/:report_logs}}
```
"""
