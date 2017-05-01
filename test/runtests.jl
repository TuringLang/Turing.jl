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
          #"opt_param_of_dist",
          "new_grammar",
          "newinterface",
          # "noreturn",
          "sample",
          "forbid_global",
#       conditional.jl
#       container.jl
          "copy_particle_container",
#       varinfo.jl
          "replay",
          "test_varname",
          "varinfo",
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
#       nuts.jl
          "nuts_cons",
          "nuts",
          # "nuts_geweke",
#       enuts.jl
          # "enuts_cons",
          # "enuts",
          # "enuts_geweke",
#       hmcda.jl
          "hmcda_cons",
          "hmcda",
          # "hmcda_geweke",
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
      # "normal_loc",
      # "normal_mixture",
      # "naive_bayes"
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



using Distributions, Turing
using ForwardDiff: Dual
using Base.Test

# Run tests in parallel
# Code adapted from runtest.jl from Distributions.jl package

print_with_color(:blue, "[runtests.jl] testing starts\n")

if nworkers() > 1
    rmprocs(workers())
end

if Base.JLOptions().code_coverage == 1
    addprocs(Sys.CPU_CORES, exeflags = ["--code-coverage=user", "--inline=no", "--check-bounds=yes"])
else
    addprocs(Sys.CPU_CORES, exeflags = "--check-bounds=yes")
end

@everywhere using Distributions, Turing
@everywhere using ForwardDiff: Dual
@everywhere using Base.Test
@everywhere srand(345679)
@everywhere testcases_v04 = ["beta_binomial", "tarray"]
@everywhere testcases_excluded = ["tarray2", "predict"]

# Run tests
# @everywhere path = dirname(@__FILE__)

res = pmap(testcases) do t
  if ~ (t in testcases_excluded)
    if t in testcases_v04
      if VERSION < v"0.5"
        println("[runtests.jl] \"$t.jl\" is running")
        include(t*".jl");
        # include(joinpath(path, t*".jl"));
        # readstring(`julia $t.jl`)
        println("[runtests.jl] \"$t.jl\" is successful")
      end
    else
      println("[runtests.jl] \"$t.jl\" is running")
      include(t*".jl");
      # include(joinpath(path, t*".jl"));
      # readstring(`julia $t.jl`)
      println("[runtests.jl] \"$t.jl\" is successful")
    end
  end
  nothing
end

println("[runtests.jl] all tests pass")
