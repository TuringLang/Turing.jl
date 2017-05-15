# Script adapted from https://github.com/stan-dev/example-models/blob/master/misc/hmm/hmm-semisup.data.R
#
# require(MCMCpack)
#
# # CONSTANTS
# K <- 3;
# V <- 10;
# T <- 100;
# T_unsup <- 500;
# alpha <- rep(1,K);
# beta <- rep(0.1,V);
#
# # DATA
# w <- rep(0,T);
# z <- rep(0,T);
# u <- rep(0,T_unsup);
#
# # PARAMETERS
# theta <- rdirichlet(K,alpha);
# phi <- rdirichlet(K,beta);
#
# # SIMULATE DATA
#
# # supervised
# z[1] <- sample(1:K,1);
# for (t in 2:T)
#   z[t] <- sample(1:K,1,replace=TRUE,theta[z[t - 1], 1:K]);
# for (t in 1:T)
#   w[t] <- sample(1:V,1,replace=TRUE,phi[z[t],1:V]);
#
# # unsupervised
# y <- rep(0,T_unsup);
# y[1] <- sample(1:K,1);
# for (t in 2:T_unsup)
#   y[t] <- sample(1:K,1,replace=TRUE,theta[y[t-1],1:K]);
# for (t in 1:T_unsup)
#   u[t] <- sample(1:V,1,replace=TRUE,phi[y[t], 1:V]);

using Distributions

# CONSTANTS
K = 3;
V = 10;
T = 100;
T_unsup = 500;
alpha = collect(repeated(1,K));
beta = collect(repeated(0.1,V));

# DATA
w = collect(repeated(0,T));
z = collect(repeated(0,T));
u = collect(repeated(0,T_unsup));

# PARAMETERS
theta = rand(Dirichlet(alpha), K);
phi = rand(Dirichlet(beta), K);

# SIMULATE DATA
theta[2,:]
# supervised
z[1] = rand(1:K);
for t in 2:T
  z[t] = rand(Categorical(theta[:,z[t - 1]]))
end
for t in 2:T
  w[t] = rand(Categorical(phi[:,z[t]]))
end

# unsupervised
y = collect(repeated(0,T_unsup));
y[1] = rand(1:K);
for t in 2:T_unsup
  y[t] = rand(Categorical(theta[:,y[t - 1]]))
end
for t in 2:T_unsup
  u[t] = rand(Categorical(phi[:,y[t]]))
end

const hmm_semisup_data = [
  Dict(
  "K" => K,
  "V" => V,
  "T" => T,
  "T_unsup" => T_unsup,
  "w" => w,
  "z" => z,
  "u" => u,
  "alpha" => alpha,
  "beta" => beta
  )
]

using HDF5, JLD

save(Pkg.dir("Turing")*"/nips-2017/hmm/hmm_semisup_data.jld", "data", hmm_semisup_data)
