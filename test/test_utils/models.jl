# The old-gdemo model.
@model function gdemo(x, y)
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
  return s, m
end

@model function gdemo_d()
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

gdemo_default = gdemo_d()

@model function MoGtest(D)
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
@model empty_model() = x = 1


# Wrapper function to quickly check gdemo accuracy.
function check_gdemo(chain; atol=0.2, rtol=0.0)
    check_numerical(chain, [:s, :m], [49/24, 7/6], atol=atol, rtol=rtol)
end

# Wrapper function to check MoGtest.
function check_MoGtest_default(chain; atol=0.2, rtol=0.0)
    check_numerical(chain,
        [:z1, :z2, :z3, :z4, :mu1, :mu2],
        [1.0, 1.0, 2.0, 2.0, 1.0, 4.0],
        atol=atol, rtol=rtol)
end

#
# Ranodm Measure related testing functions
#

function compute_log_joint(observations, partition, tau0, tau1, sigma, theta)
    n = length(observations)
    k = length(partition)
    prob = k*log(sigma) + lgamma(theta) + lgamma(theta/sigma + k) - lgamma(theta/sigma) - lgamma(theta + n)
    for cluster in partition
      prob += lgamma(length(cluster) - sigma) - lgamma(1 - sigma)
      prob += compute_log_conditional_observations(observations, cluster, tau0, tau1)
    end
    prob
  end
  
  function compute_log_conditional_observations(observations, cluster, tau0, tau1)
    nl = length(cluster)
    prob = (nl/2)*log(tau1) - (nl/2)*log(2*pi) + 0.5*log(tau0) + 0.5*log(tau0+nl)
    prob += -tau1/2*(sum(observations)) + 0.5*(tau0*mu_0+tau1*sum(observations[cluster]))^2/(tau0+nl*tau1)
    prob
  end
  
  # Test of similarity between distributions
  function correct_posterior(empirical_probs, data, partitions, τ0, τ1, σ, θ)
      true_log_probs = map(p -> compute_log_joint(data, p, τ0, τ1, σ, θ), partitions)
      true_probs = exp.(true_log_probs)
      true_probs /= sum(true_probs)
  
      empirical_probs /= sum(empirical_probs)
  
      # compare distribitions
      # L2
      L2 = sum((empirical_probs - true_probs).^2)
  
      # Discrepancy
      discr = maximum(abs.(empirical_probs - true_probs))
      return L2, discr
  end