##########################################
# Master test file for running all tests #
##########################################

testcases = [
    "assume",
    "beta-binomial",
    "importance_sampling",
    "noparam",
    "observe",
    "predict",
    "resample"]

res = pmap(testcases) do t
    include(t*".jl")
    nothing
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
