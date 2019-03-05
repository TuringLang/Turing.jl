### Compute theoretical posterior distribution over partitions
# cf Maria Lomeli's thesis, Section 2.7.4, page 46.

function compute_log_joint(observations, partition, tau0, tau1)
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

true_log_probs = [compute_log_joint(data, partition, tau0, tau1) for partition in Partitions]
true_probs = exp.(true_log_probs)
true_probs /= sum(true_probs)
