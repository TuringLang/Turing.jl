using Dates

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

run(`git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'`)
run(`git fetch --all --unshallow`)

run(`git clone https://github.com/TuringLang/TuringBenchmarks.git ../TuringBenchmarks`)

delete!(ENV, "JULIA_PROJECT")

code_pre = """using Pkg
# Pkg.instantiate()
try pkg"develop ." catch end
try pkg"develop ." catch end
try pkg"build Turing" catch end
using Turing
try pkg"develop ../TuringBenchmarks" catch end
try pkg"develop ../TuringBenchmarks" catch end
pkg"add SpecialFunctions"
using TuringBenchmarks
"""

code_run = """using TuringBenchmarks.Runner
Runner.run_bm_on_travis("$BM_JOB_NAME", ("master", "$CURRENT_BRANCH"), "$COMMIT_SHA")
"""

run(`julia -e $code_pre`)
run(`julia -e $code_run`)
