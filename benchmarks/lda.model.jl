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

# TODO: improve the model
@model ldamodel(K, V, M, N, w, doc, alpha, β) = begin
  theta = Array{Any}(M)
  for m = 1:M
    theta[m] ~ Dirichlet(alpha)
  end
  phi = Array{Any}(K)
  for k = 1:K
    phi[k] ~ Dirichlet(β)
  end
  z = Array{Vector{Int}}(N)
  map!(x -> Vector{Int}(M), z)
  for n = 1:N, m = 1:M
    z[n][m] ~ Categorical(theta[m])
  end
  for n = 1:N
    w[n] ~ Categorical(phi[z[n][doc[n]]])
  end
end
