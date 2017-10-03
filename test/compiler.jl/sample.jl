using Distributions
using Turing

@model gdemo2(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i = 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  return(s, m, x)
end

x = [2.0, 3.0]

alg = Gibbs(1000, HMC(1, 0.2, 3, :m), PG(10, 1, :s))
# NOTE: want to translate below to
#       chn = sample(gdemo, Dict(:x => x), alg)
modelf = gdemo2(x)
chn = sample(modelf, alg);
mean(chn[:s])

# Turing.TURING[:modelex]
