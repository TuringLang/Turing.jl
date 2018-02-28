

using Distributions
using Turing
using StatsFuns: logsumexp

@model hmm_semisup(K,V,T,T_unsup,w,z,u,alpha,beta) = begin
  theta = Vector{Vector{Real}}(K)
  for k = 1:K
    theta[k] ~ Dirichlet(alpha)
  end
  phi = Vector{Vector{Real}}(K)
  for k = 1:K
    phi[k] ~ Dirichlet(beta)
  end

  for t = 1:T
    w[t] ~ Categorical(phi[z[t]])
  end
  for t = 2:T
    z[t] ~ Categorical(theta[z[t-1]])
  end

  if collapsed
    acc = Vector{Real}(K)
    gamma = Matrix{Real}(T_unsup,K)
    for k = 1:K
      gamma[1,k] = log(phi[k][u[1]])
    end
    for t = 2:T_unsup,
        k = 1:K
        for j = 1:K
          acc[j] = gamma[t-1,j] + log(theta[j][k]) + log(phi[k][u[t]])
        end
        gamma[t,k] = logsumexp(acc)
    end
    Turing.acclogp!(vi, logsumexp(gamma[T_unsup,:]))
  else
    y = tzeros(Int64, T_unsup)
    y[1] ~ Categorical(ones(Float64, K)/K)
    u[1] ~ Categorical(phi[y[1]])
    for t in 2:T_unsup
      y[t] ~ Categorical(theta[y[t - 1]])
      u[t] ~ Categorical(phi[y[t]])
    end

  end
end
