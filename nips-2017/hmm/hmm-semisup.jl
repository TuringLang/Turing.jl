const hmm_semisup_stan = "
data {
  int<lower=1> K;  // num categories
  int<lower=1> V;  // num words
  int<lower=0> T;  // num supervised items
  int<lower=1> T_unsup;  // num unsupervised items
  int<lower=1,upper=V> w[T]; // words
  int<lower=1,upper=K> z[T]; // categories
  int<lower=1,upper=V> u[T_unsup]; // unsup words
  vector<lower=0>[K] alpha;  // transit prior
  vector<lower=0>[V] beta;   // emit prior
}
parameters {
  simplex[K] theta[K];  // transit probs
  simplex[V] phi[K];    // emit probs
}
model {
  for (k in 1:K)
    theta[k] ~ dirichlet(alpha);
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);
  for (t in 1:T)
    w[t] ~ categorical(phi[z[t]]);
  for (t in 2:T)
    z[t] ~ categorical(theta[z[t-1]]);

  {
    // forward algorithm computes log p(u|...)
    real acc[K];
    real gamma[T_unsup,K];
    for (k in 1:K)
      gamma[1,k] <- log(phi[k,u[1]]);
    for (t in 2:T_unsup) {
      for (k in 1:K) {
        for (j in 1:K)
          acc[j] <- gamma[t-1,j] + log(theta[j,k]) + log(phi[k,u[t]]);
        gamma[t,k] <- log_sum_exp(acc);
      }
    }
    increment_log_prob(log_sum_exp(gamma[T_unsup]));
  }
}
"

using HDF5, JLD
const hmm_semisup_data = load(Pkg.dir("Turing")*"/nips-2017/hmm/hmm_semisup_data.jld")["data"]

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

  acc = Vector{Float64}(K)
  gamma = Matrix{Float64}(T_unsup,K)
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
  Turing.acclogp!(logsumexp(gamma[T_unsup,:]))
end

sample(hmm_semisup(data=hmm_semisup_data[1]), NUTS(200, 0.65))
