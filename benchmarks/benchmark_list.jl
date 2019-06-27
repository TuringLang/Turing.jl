BENCHMARK_FILES = [
    "dummy.run.jl",
    "gdemo.run.jl",
    "mvnormal.run.jl",
]

BENCHMARK_FILES = map(BENCHMARK_FILES) do x joinpath(@__DIR__, x) end
