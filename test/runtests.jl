##########################################
# Master file for running all test cases #
##########################################

# NOTE: please keep this test list structured when adding new test cases
# so that we can tell which test case is for which .jl file

testcases = [
# Turing.jl/
#   src/
#     core/
#       ad.jl
          "ad",
          "ad2",
          "pass_dual_to_dists",
#       compiler.jl
          "assume",
          "observe",
          "predict",
          "beta_binomial",
          "noparam",
          "opt_param_of_dist",
          "new_grammar",
          "newinterface",
#       conditional.jl
#       container.jl
          "copy_particle_container",
#       graidnetinfo.jl
          "replay",
          "gradientinfo",
          "flaten_naming",
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
          "transform",
#     samplers/
#       support/
#         resample.jl
            "resample",
            "particlecontainer",
#       gibbs.jl
            "gibbs",
            "gibbs2",
            "gibbs_constructor",
#       hmc.jl
          "multivariate_support",
          "matrix_support",
          "constrained_bounded",
          "constrained_simplex",
#       is.jl
          "importance_sampling",
#       pgibbs.jl
#       sampler.jl
#       smc.jl
#     trace/
#       tarray.jl
          "tarray",
          "tarray2",
          "tarray3",
#       taskcopy.jl
          "clonetask",
#       trace.jl
          "trace",
#   Turing.jl
      "normal_loc",
      "normal_mixture",
      "naive_bayes"
# NOTE: not comma for the last element
]

# NOTE: put test cases which only want to be check in version 0.4.x here
testcases_v04 = [
  "beta_binomial",
  "tarray"
]

# NOTE: put test cases which want to be excluded here
testcases_excluded = [
  "tarray2",
  "predict"
]

# Run tests
path = dirname(@__FILE__)
cd(path)
println("[runtests.jl] testing starts")
for t in testcases
  if ~ (t in testcases_excluded)
    if t in testcases_v04
      if VERSION < v"0.5"
        println("[runtests.jl] \"$t.jl\" is running")
        include(t*".jl");
        # readstring(`julia $t.jl`)
        println("[runtests.jl] \"$t.jl\" is successful")
      end
    else
      println("[runtests.jl] \"$t.jl\" is running")
      include(t*".jl");
      # readstring(`julia $t.jl`)
      println("[runtests.jl] \"$t.jl\" is successful")
    end
  end
end
println("[runtests.jl] all tests pass")
