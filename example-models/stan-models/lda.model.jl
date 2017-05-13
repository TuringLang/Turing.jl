# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/misc/cluster/lda/lda.stan

# data {
#   int<lower=2> K;               // num topics
#   int<lower=2> V;               // num words
#   int<lower=1> M;               // num docs
#   int<lower=1> N;               // total word instances
#   int<lower=1,upper=V> w[N];    // word n
#   int<lower=1,upper=M> doc[N];  // doc ID for word n
#   vector<lower=0>[K] alpha;     // topic prior
#   vector<lower=0>[V] beta;      // word prior
# }
# parameters {
#   simplex[K] theta[M];   // topic dist for doc m
#   simplex[V] phi[K];     // word dist for topic k
# }
# model {
#   for (m in 1:M)
#     theta[m] ~ dirichlet(alpha);  // prior
#   for (k in 1:K)
#     phi[k] ~ dirichlet(beta);     // prior
#   for (n in 1:N) {
#     real gamma[K];
#     for (k in 1:K)
#       gamma[k] <- log(theta[doc[n],k]) + log(phi[k,w[n]]);
#     increment_log_prob(log_sum_exp(gamma));  // likelihood
#   }
# }

@model ldamodel(K, V, M, N, w, doc, beta, alpha) = begin
  theta = Vector{Vector{Real}}(M)
  for m = 1:M
    theta[m] ~ Dirichlet(alpha)
  end

  phi = Vector{Vector{Real}}(K)
  for k = 1:K
    phi[k] ~ Dirichlet(beta)
  end

  # z = tzeros(Int, N)
  # for n = 1:N
  #   z[n] ~ Categorical(theta[doc[n]])
  # end

  phi_dot_theta = [log([dot(map(p -> p[i], phi), theta[m]) for i = 1:V]) for m=1:M]
  for n = 1:N
    # phi_dot_theta = [dot(map(p -> p[i], phi), theta[doc[n]]) for i = 1:V]
    # w[n] ~ Categorical(phi_dot_theta)
    Turing.acclogp!(vi, phi_dot_theta[doc[n]][w[n]])
  end

end
