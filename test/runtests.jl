##########################################
# Master file for running all test cases #
##########################################

testcases = [
    "assume",
    "beta-binomial",
    "importance_sampling",
    "noparam",
    "observe",
    "predict",
    "resample"]

for t in testcases include(t*".jl") end

testcases_v05 = [
    "beta-binomial",
    "test_tarray"]

if VERSION < v"0.5"
  for t in testcases_v05 include(t*".jl") end
end


include("test_clonetask.jl")
include("test_tarray.jl")

# include("test_tarray2.jl")
include("test_particlecontainer.jl")

# For HMC
include("test_priorcontainer.jl")
include("test_replay.jl")
include("test_multidimsupport.jl")
include("test_distributions_with_dual.jl")
