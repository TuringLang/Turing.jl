# MicroBenchmarks for Turing.jl

## Write a benchmark

Generally, you can run any code in a benchmark. We use package
`BenchmarkTools.jl` to organize our benchmarks, and
`BenchmarkHelper.BenchmarkSuite` is the root benchmark group, so you
can write a benchmark like this:

```julia
using Turing, BenchmarkHelper, BenchmarkTools

BenchmarkSuite["dummy"] = BenchmarkGroup(["tag1", "tag2"])

BenchmarkSuite["dummy"]["benchmark_1"] = @benchmarkable a_time_consuming_computing()
```
More details can be found in the documents of `BenchmarkTools.jl`.

After its run, the results will be saved under the directory
`benchmarks/output/`.

## Run on Github

All benchmarks under this directory can run on Github, triggered by
the following event:

- Comment on an issue (This is disable if we use Nanosoldier.jl to run
  the action)
- Comment on a Pull Request (a.k.a Code Review Comment)

You must **@BayesBot** and include a command in pattern
`runbenchmarks(benchamrk_tags, vs=branch)` in the comment text to
specify the branches on which the benchmarks will run and trigger the
benchmarks. This command pattern is borrowed from `Nanosoldier.jl`,
please find more detail from its documents. Here are some examples:

- `@BayesBot runbenchmarks("array", vs=[“:br1”, ":br2"])` will trigger
    benchmarks tagged with `"array"` on branch `br1` and `br2`
- Comments on a regular issue
  - `@BayesBot runbenchmarks("array")` won't trigger any benchmarks due to no
    branches are specified
  - `@BayesBot runbenchmarks("tag2", vs=":feature-1")` will trigger benchmarks
    tagged with `"tag2"` on branch `master` and `feature-1`
- Comments on a pull request
  - `@BayesBot runbenchmarks(ALL)` will trigger all benchmarks on the pull
    request branch and the `master` branch
  - `@BayesBot runbenchmark(ALL, vs=":feature-1")` will trigger all benchmarks
    on the pull request branch and the `:feature-1` branch


After these triggered benchmarks finish, a reply comment will be made
to report the results, or a commit will be made to the
TuringLang/BenchmarkReports repo if Nanodoldier is used.

### Notes

- Put a Regex pattern to the `EXCLUDES` array in
  `benchmarks/config.toml` to exclude paricular benchmarks

## Run on local machine

We can use `benchmarks/local-runner` to run a benchmark on our local machine:

```bash
./benchmarks/local-runner benchmarks/benchmarks.jl
```
