using Turing

@model gaussdemo begin
  @assume s ~ InverseGamma(2,3)
  @assume m ~ Normal(0,sqrt(s))
  @observe 1.5 ~ Normal(m, sqrt(s))
  @observe 2.0 ~ Normal(m, sqrt(s))
  @predict s m
end

# Sample and print.
res = sample(gaussdemo, SMC(10000))
println("Infered: m = $(mean(res[:m])), s = $(mean(res[:s]))")

# Compute analytical solution. Requires `ConjugatePriors` package.
exact = posterior(NormalInverseGamma(0,1,2,3), Normal, [1.5,2.0])
println("Exact: m = $(mean(exact)[1]), s = $(mean(exact)[2])")
