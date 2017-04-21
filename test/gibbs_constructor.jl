using Turing, Distributions
using Turing: GibbsSampler
using Base.Test

@model gdemo() = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

s1 = Gibbs(1000, HMC(10, 0.1, 5, :s, :m))
s2 = Gibbs(1000, PG(10, 10, :s, :m))
s3 = Gibbs(1000, PG(10, 2, :s), HMC(1, 0.4, 8, :m))
s4 = Gibbs(1000, PG(10, 3, :s), HMC(2, 0.4, 8, :m); thin=false)


c1 = sample(gdemo(), s1)
c2 = sample(gdemo(), s2)
c3 = sample(gdemo(), s3)
c4 = sample(gdemo(), s4)

# Very loose bound, only for testing constructor.
@test_approx_eq_eps mean(c1[:s]) 49/24 1
@test_approx_eq_eps mean(c1[:m]) 7/6 1
@test_approx_eq_eps mean(c2[:s]) 49/24 1
@test_approx_eq_eps mean(c2[:m]) 7/6 1
@test_approx_eq_eps mean(c3[:s]) 49/24 1
@test_approx_eq_eps mean(c3[:m]) 7/6 1

@test length(c4[:s]) == 1000 * (3 + 2)
@test_approx_eq_eps mean(c4[:s]) 49/24 1
@test_approx_eq_eps mean(c4[:m]) 7/6 1


# Test group_id of each samplers
g = GibbsSampler{Gibbs}(s3)

@test g.samplers[1].alg.group_id == 1
@test g.samplers[2].alg.group_id == 2
