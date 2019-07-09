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
