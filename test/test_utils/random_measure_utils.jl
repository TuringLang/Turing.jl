using SpecialFunctions

function compute_log_joint(observations, partition, tau0, tau1, sigma, theta)
  n = length(observations)
  k = length(partition)
  prob = k*log(sigma) + lgamma(theta) + lgamma(theta/sigma + k) - lgamma(theta/sigma) - lgamma(theta + n)
  for cluster in partition
    prob += lgamma(length(cluster) - sigma) - lgamma(1 - sigma)
    prob += compute_log_conditonal_observations(observations, cluster, tau0, tau1)
  end
  prob
end

function compute_log_conditonal_observations(observations, cluster, tau0, tau1)
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
