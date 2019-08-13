CONFIG = Dict(
    "target.project_dir" => (@__DIR__) |> dirname |> abspath,
    "reporter.use_dataframe" => true,
)

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

CONFIG["benchmark_files"] = map(BENCHMARK_FILES) do x joinpath(@__DIR__, x) end
CONFIG["comment_templates.ci_bm_complete_for_commit"] = """

The benchmark job for this commit has completed. 
Here is the benchmark [report]( {{{ :report_url }}}) and commit ({{{ :report_repo }}}@{{{ :commit_id }}}). 
The benchmark log (printed via `log_report`) can be found below.

```
{{#:report_logs}}
{{.}}
{{/:report_logs}}
```
"""
