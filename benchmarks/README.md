# Benchmarks for Turing.jl

## Write a benchmark

Generally, you can run any code in a benchmark. We use package
`BenchmarkTools.jl` to organize our benchmarks, and
`BenchmarkSuite` is the root benchmark group, so you
can write a benchmark like this:

```julia
using Turing, BenchmarkTools

BenchmarkSuite["example_benchmark"] = BenchmarkGroup(["tag1", "tag2"])

BenchmarkSuite["example_benchmark"]["benchmark_1"] = @benchmarkable a_time_consuming_computing()
```

More details can be found in the documents of `BenchmarkTools.jl`.
