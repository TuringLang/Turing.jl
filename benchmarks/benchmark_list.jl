ON_TRAVIS = get(ENV, "TRAVIS", "false") == "true"

if ON_TRAVIS
    BENCHMARK_FILES = [
        "dummy.run.jl",
        "gdemo.run.jl",
        "mvnormal.run.jl",
    ]
else
    BENCHMARK_FILES = [
        "dummy.run.jl",
        "gdemo.run.jl",
        "mvnormal.run.jl",
    ]
end

BENCHMARK_FILES = map(BENCHMARK_FILES) do x joinpath(@__DIR__, x) end
