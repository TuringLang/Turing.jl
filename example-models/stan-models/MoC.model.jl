# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/misc/cluster/naive-bayes/naive-bayes.stan
# Reference: http://mlg.eng.cam.ac.uk/teaching/4f13/1617/document%20models.pdf
@model nbmodel(K, V, M, N, z, w, doc, alpha, beta) = begin
  theta ~ Dirichlet(alpha)
  phi = Array{Any}(K)
  for k = 1:K
    phi[k] ~ Dirichlet(beta)
  end
  #for m = 1:M
  #  z[m] ~ Categorical(theta)
  #end

  log_theta = log.(theta)
  Turing.acclogp!(vi, sum(log_theta[z]))

  log_phi = map(x->log.(x), phi)
  _lp = mapreduce(n->log_phi[z[doc[n]]][w[n]], +, 1:N)
  # Turing.acclogp!(vi, lp)

  phi
end
