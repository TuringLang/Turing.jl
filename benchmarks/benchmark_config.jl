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

The benchmark job for this commit has completed. 
Here is the benchmark [report]( {{{ :report_url }}}) and commit ({{{ :report_repo }}}@{{{ :commit_id }}}). 
The benchmark log (printed via `log_report`) can be found below.

```
{{#:report_logs}}
{{.}}
{{/:report_logs}}
```
"""
