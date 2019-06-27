BENCHMARK_FILES = [
    "Dummy.run.jl",
]

BENCHMARK_FILES = map(BENCHMARK_FILES) do x joinpath(@__DIR__, x) end
