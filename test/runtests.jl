##########################################
# Master file for running all test cases #
##########################################

# NOTE: please keep this test list structured when adding new test cases
# so that we can tell which test case is for which .jl file

testcases = [
# Turing.jl/
#   src/
#     core/
#       compiler.jl
          "assume",
          "observe",
          "predict",
          "beta_binomial",
          "noparam",
          "opt_param_of_dist",
#       conditional.jl
#       container.jl
          "copy_particle_container",
#       IArray.jl
#       intrinsic.jl
#       io.jl
          "chain_utility",
#       util.jl
          "util",
#     distributions/
#       bnp.jl
#       distributions.jl
#       transform.jl
#     samplers/
#       support/
#         reply.jl
            "replay",
            "priorcontainer",
#         resample.jl
            "resample",
            "particlecontainer",
#       hmc.jl
          "pass_dual_to_dists",
          "multivariate_support_for_hmc",
#       is.jl
          "importance_sampling",
#       pgibbs.jl
#       sampler.jl
#       smc.jl
#     trace/
#       tarray.jl
          "tarray",
          "tarray2",
#       taskcopy.jl
          "clonetask",
#       trace.jl
          "trace"
# NOTE: not comma for the last element
]

# NOTE: put test cases which only want to be check in version 0.4.x here
testcases_v04 = [
  "beta_binomial",
  "tarray"
]

# NOTE: put test cases which want to be excluded here
testcases_excluded = [
  "tarray2"
]

# Run tests
println("[runtests.jl] testing starts")
for t in testcases
  if ~ (t in testcases_excluded)

    if t in testcases_v04
      if VERSION < v"0.5"
        println("[runtests.jl] running test \"$t.jl\"")
        include(t*".jl")
        println("[runtests.jl] test \"$t.jl\" is successful")
      end
    else
      println("[runtests.jl] running test \"$t.jl\"")
      include(t*".jl")
      println("[runtests.jl] test \"$t.jl\" is successful")
    end
  end
end
println("[runtests.jl] all tests pass")
