##########################################
# Master file for running all test cases #
##########################################
using Turing; turnprogress(false)
using Test

# utility function
function getteststoskip(filepath)
  if isfile(filepath)
    lines = readlines(filepath)
    lines = filter(line -> endswith(line, ".jl"), lines)
    lines = filter(line -> !startswith(line, "#"), lines)
    lines = filter(line -> length(split(line)) == 1, lines)
    lines = map(line -> strip(line), lines)
    return Set{String}(lines)
  else
    return Set{String}()
  end
end

@info("[runtests.jl] runtests.jl loaded")

# THIS SHOULDE BE PORTED ??
testcases_excluded = [
  "tarray2",
  "predict"
]

# test groups
CORE_TESTS = ["ad.jl", "compiler.jl", "container.jl", "varinfo.jl",
    # "io.jl",
    "util.jl"]
DISTR_TESTS = ["transform.jl"]
SAMPLER_TESTS = ["resample.jl", "adapt.jl", "vectorisation.jl", "gibbs.jl", "nuts.jl",
                 "hmcda.jl", "hmc_core.jl", "hmc.jl", "sghmc.jl", "sgld.jl", "is.jl",
                 "mh.jl", "pmmh.jl", "ipmcmc.jl",
                # "pmmh.jl", "pgibbs.jl", "smc.jl"
                ]
TRACE_TESTS = ["tarray.jl", "taskcopy.jl", "trace.jl"]
ALL = union(CORE_TESTS, DISTR_TESTS, SAMPLER_TESTS, TRACE_TESTS)

# test groups that should be executed
TEST_GROUPS = ALL
# TEST_GROUPS = ["compiler.jl"]

# Run tests
path = dirname(@__FILE__)
cd(path); include("utility.jl")
@info("[runtests.jl] utility.jl loaded")
@info("[runtests.jl] testing starts")
for test_group in TEST_GROUPS
  teststoskip = getteststoskip(joinpath(test_group, "skip_tests"))
  @testset "$(test_group)" begin
    for test in filter(f -> endswith(f, ".jl") && !(f ∈ teststoskip), readdir(test_group))
      @testset "$(test)" begin
        include(joinpath(test_group, test))
      end
    end
  end
end

@info("[runtests.jl] all tests finished")
