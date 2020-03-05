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
commenting on a Pull Request (a.k.a Code Review Comment).

You must **@BayesBot** and include a command in pattern
`runbenchmarks(benchamrk_tags, vs=branch)` in the comment text to
specify the branches on which the benchmarks will run and trigger the
benchmarks. This command pattern is borrowed from `Nanosoldier.jl`,
please find more detail from [its
documents](https://github.com/TuringLang/Nanosoldier.jl/blob/turing/README.md#trigger-syntax). Here
is an examples:

```
@BayesBot `runbenchmarks(ALL, vs=":master")`
```

After these triggered benchmarks finish, the reports will be committed
to
[TuringLang/BenchmarkReports](https://github.com/TuringLang/BenchmarkReports),
then a reply comment contains the reports link will be made on the
original pull request.

### Notes

- Put a Regex pattern to the `EXCLUDES` array in
  `benchmarks/config.toml` to exclude paricular benchmark files
