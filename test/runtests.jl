##########################################
# Master file for running all test cases #
##########################################

testcases = [
  "assume",
  "beta_binomial",
  "importance_sampling",
  "noparam",
  "observe",
  "predict",
  "resample",
  "clonetask",
  "particlecontainer",
  "priorcontainer",
  "trace",
  "multivariate_support_for_hmc",
  "pass_dual_to_dists"
]

testcases_v05 = [
  "beta_binomial",
  "tarray"
]

testcases_untouched = [
  "tarray2"
]

for t in testcases include(t*".jl") end

if VERSION < v"0.5"
  for t in testcases_v05 include(t*".jl") end
end
