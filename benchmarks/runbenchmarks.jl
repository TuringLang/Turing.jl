using Pkg
using Dates

PROJECT_DIR = abspath(@__DIR__) |> dirname

# prepare packages
Pkg.build("Turing")

BENCHMARK_REV = "master"
Pkg.add(PackageSpec(url="https://github.com/TuringLang/TuringBenchmarks.git", rev=BENCHMARK_REV))
Pkg.build("TuringBenchmarks")
Pkg.resolve()

# prepare BenchMark information
BASE_BRANCH = "master"
CURRENT_BRANCH = strip(read(`git rev-parse --abbrev-ref HEAD`, String))

if get(ENV, "TRAVIS", "false") == "true"
    if get(ENV, "TRAVIS_PULL_REQUEST", "false") == "true"
        CURRENT_BRANCH = get(ENV, "TRAVIS_PULL_REQUEST_BRANCH")
        exit(0)
    else
        CURRENT_BRANCH = get(ENV, "TRAVIS_BRANCH", "master")
    end
end

SANTI_BR_NAME = 0
COMMIT_SHA = strip(read(`git rev-parse  HEAD`, String))
COMMIT_SHA_7 = COMMIT_SHA[1:7]
TIME = Dates.format(now(), "YYYYmmddHHMM")
BM_JOB_NAME="BMCI-$(SANTI_BR_NAME)-$(COMMIT_SHA_7)-$(TIME)"

if get(ENV, "TRAVIS", "false") == "true"
    run(`git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'`)
    run(`git fetch --all --unshallow`)
end

# run
code_run = """using TuringBenchmarks
using TuringBenchmarks.Runner
TuringBenchmarks.set_project_path("$PROJECT_DIR")
TuringBenchmarks.set_benchmark_files(joinpath("$PROJECT_DIR", "benchmarks/benchmark_list.jl"))
Runner.run_bm_on_travis("$BM_JOB_NAME", ("$BASE_BRANCH", "$CURRENT_BRANCH"), "$COMMIT_SHA")
"""
run(`julia --project=$PROJECT_DIR -e $code_run`)
