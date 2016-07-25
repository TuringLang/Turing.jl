function ess(samples)
  """
  @input:
    samples : a vector of samples
  @output:
    ess = n / (1 + 2∑ρ)
  """
  n = length(samples)
  acfs = StatsBase.autocor(samples, [1:(n - 1), deman=false])
  # Truncate the array according to http://search.r-project.org/library/LaplacesDemon/html/ESS.html
  acfs_trancated = []
  for i = 1:n
    if acfs[i] < 0.05
      break
    end
    push!(acfs_trancated, acfs[i])
  end
  # filter!(a -> a > 0.05, acfs)
  return n / (1 + 2 * sum(acfs_trancated))
end
