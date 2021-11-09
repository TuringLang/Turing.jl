# The old-gdemo model.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
  return s, m
end

@model gdemo_d() = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

gdemo_default = gdemo_d()

@model MoGtest(D) = begin
    mu1 ~ Normal(1, 1)
    mu2 ~ Normal(4, 1)
    z1 ~ Categorical(2)
    if z1 == 1
        D[1] ~ Normal(mu1, 1)
    else
        D[1] ~ Normal(mu2, 1)
    end
    z2 ~ Categorical(2)
    if z2 == 1
        D[2] ~ Normal(mu1, 1)
    else
        D[2] ~ Normal(mu2, 1)
    end
    z3 ~ Categorical(2)
    if z3 == 1
        D[3] ~ Normal(mu1, 1)
    else
        D[3] ~ Normal(mu2, 1)
    end
    z4 ~ Categorical(2)
    if z4 == 1
        D[4] ~ Normal(mu1, 1)
    else
        D[4] ~ Normal(mu2, 1)
    end
    z1, z2, z3, z4, mu1, mu2
end

MoGtest_default = MoGtest([1.0 1.0 4.0 4.0])

# Declare empty model to make the Sampler constructor work.
@model empty_model() = begin x = 1; end
