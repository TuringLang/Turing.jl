using Turing
using Test

@model gdemo() = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

N = 500
s1 = Gibbs(N, HMC(10, 0.1, 5, :s, :m))
s2 = Gibbs(N, PG(10, 10, :s, :m))
s3 = Gibbs(N, PG(10, 2, :s), HMC(1, 0.4, 8, :m))
s4 = Gibbs(N, PG(10, 3, :s), HMC(2, 0.4, 8, :m); thin=false)
s5 = Gibbs(N, CSMC(10, 2, :s), HMC(1, 0.4, 8, :m))


c1 = sample(gdemo(), s1)
c2 = sample(gdemo(), s2)
c3 = sample(gdemo(), s3)
c4 = sample(gdemo(), s4)
c5 = sample(gdemo(), s5)

# Very loose bound, only for testing constructor.
## Note: undo this and check `check_numerical`
#for c in [c1, c2, c3 ,c4, c5]
#  check_numerical(c, [:s, :m], [49/24, 7/6], eps=1.0)
#end

@test length(c4[:s]) == N * (3 + 2)

# Test gid of each samplers
g = Turing.Sampler(s3, gdemo())

@test g.info[:samplers][1].alg.gid == 1
@test g.info[:samplers][2].alg.gid == 2
