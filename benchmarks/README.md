# MicroBenchmarks for Turing.jl

## Write a benchmark

Generally, you can run any code in a benchmark. In order to collect
information from benchmarks, we define a simple protocol which
includes:

- Function `BenchmarkHelper.log_report(log::String)`, which is used to
  report logs in the running process
- A global varibale, `BENCHMARK_RESULT`, which is a `Dict{Any, Any}`
  and used to store the result of the benchmark, like benchmark ID,
  time used, memory consumed, etc.

Here is a simple example:

```julia
using Turing, BenchmarkHelper

log_report("I want to report a log here.")

result, t_elapsed, mem, gctime, memallocs  = @timed an_expensive_funcall()

BENCHMARK_RESULT = Dict(
            "engine" => "A Simple Example",
            "time" => t_elapsed,
            "memory" => mem,
        )

log_report("Another log: Benchmark finished!")

```

After its run, the results (the reported logs and the
`BENCHMARK_RESULT`) will be saved under the directory
`benchmarks/output/`.

## Run on Github

All benchmarks under this directory can run on Github, triggered by
the following event:

- Comment on an issue
- Comment on a Pull Request (a.k.a Code Review Comment)

You must include a command in pattern `!benchmark(...)` in the comment
text to specify the branches on which the benchmarks will run and
trigger the benchmarks. Here are some examples:

- `!benchmark(feature-1, feature-2)` will trigger all benchmarks on
    branch `feature-1` and `feature-2`
- Comment on a regular issue
  - `!benchmark()` won't trigger any benchmarks due to no branches are
    specified
  - `!benchmark(feature-1)` or `!benchmark(master, feature-1)`
    will trigger all benchmarks on branch `master` and `feature-1`
- Comment on a pull request
  - `!benchmark()` will trigger all benchmarks on the pull request
    branch and the `master` branch
  - `!benchmark(feature-1)` will trigger all benchmarks on the pull request
    branch and the `feature-1` branch


After these triggered benchmarks finish, a reply comment will be made
to report the results.

### Notes

- Put a Regex pattern to the `EXCLUDES` array in
  `benchmarks/config.toml` to exclude paricular benchmarks
- One can use an existing Git commit ID in lieu of a branch name in
  the `!benchmark` command

## Run on local machine

We can use `benchmarks/local-runner` to run a benchmark on our local machine:

```bash
./benchmarks/local-runner benchmarks/dummy.jl
```
