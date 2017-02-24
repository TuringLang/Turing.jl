# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/misc/cluster/naive-bayes/naive-bayes.stan

@model nbmodel begin
  theta ~ Dirichlet(alpha)
  phi = Array{Any}(K)
  for k = 1:K
    phi[k] ~ Dirichlet(β)
  end
  for m = 1:M
    z[m] ~ Categorical(theta)
  end
  for n = 1:N
    w[n] ~ Categorical(phi[z[doc[n]]])
  end
  phi
end
