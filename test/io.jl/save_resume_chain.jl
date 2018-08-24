using Turing
using Test
using Random

include("../utility.jl")

Random.seed!(123)

alg1 = HMCDA(3000, 1000, 0.65, 0.15)
alg2 = PG(20, 500)
alg3 = Gibbs(500, PG(30, 10, :s), HMCDA(1, 500, 0.65, 0.05, :m))

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt.(s))
  x[1] ~ Normal(m, sqrt.(s))
  x[2] ~ Normal(m, sqrt.(s))
  return s, m
end

chn1 = sample(gdemo([1.5, 2.0]), alg1; save_state=true)

check_numerical(chn1, [:s, :m], [49/24, 7/6])

chn1_resumed = resume(chn1, 1000)

check_numerical(chn1_resumed, [:s, :m], [49/24, 7/6])

chn1_contd = sample(gdemo([1.5, 2.0]), alg1; resume_from=chn1)

check_numerical(chn1_contd, [:s, :m], [49/24, 7/6])

chn1_contd2 = sample(gdemo([1.5, 2.0]), alg1; resume_from=chn1, reuse_spl_n=1000)

check_numerical(chn1_contd2, [:s, :m], [49/24, 7/6])

chn2 = sample(gdemo([1.5, 2.0]), alg2; save_state=true)

check_numerical(chn2, [:s, :m], [49/24, 7/6])

chn2_contd = sample(gdemo([1.5, 2.0]), alg2; resume_from=chn2)

check_numerical(chn2_contd, [:s, :m], [49/24, 7/6])

chn3 = sample(gdemo([1.5, 2.0]), alg3; save_state=true)

check_numerical(chn3, [:s, :m], [49/24, 7/6])

chn3_contd = sample(gdemo([1.5, 2.0]), alg3; resume_from=chn3)

check_numerical(chn3_contd, [:s, :m], [49/24, 7/6])
