using Distributions
using Turing

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i=1:length(x)
    x[i] ~ Normal(m, sqrt(s)) # Note: we need to fetch x from gdemo(; data = Data(...)).
  end
  return(s, m, x)
end

x = [2.0, 3.0]
alg = Gibbs(HMC(1, 0.2, 3, :m), PG(10, 1, :s))
chn = @sample(gdemo(x), alg)
