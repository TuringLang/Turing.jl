using Distributions

# V <- 5; # words: river, stream, bank, money, loan
# K <- 2; # topics: RIVER, BANK
# M <- 25;  # docs
V = 35
K = 2
M = 25

# alpha <- rep(1/K,K);
# beta <- rep(1/V,V);
alpha = collect(repeated(1/K, K));
beta = collect(repeated(1/V, V));

# phi <- array(NA,c(2,5));
# phi[1,] = c(0.330, 0.330, 0.330, 0.005, 0.005);
# phi[2,] = c(0.005, 0.005, 0.330, 0.330, 0.330);
phi = rand(Dirichlet(beta), K)

# theta <- rdirichlet(M,alpha);
theta = rand(Dirichlet(alpha), M)

# avg_doc_length <- 10;
avg_doc_length = 20

# doc_length <- rpois(M,avg_doc_length);
doc_length = rand(Poisson(avg_doc_length), M)

# N <- sum(doc_length);
N = sum(doc_length)

# w <- rep(NA,N);
# doc <- rep(NA,N);
w = Vector{Int}(N)
doc = Vector{Int}(N)

# n <- 1;
# for (m in 1:M) {
#   for (i in 1:doc_length[m]) {
#     z <- which(rmultinom(1,1,theta[m,]) == 1);
#     w[n] <- which(rmultinom(1,1,phi[z,]) == 1);
#     doc[n] <- m;
#     n <- n + 1;
#   }
# }
n = 1
for m = 1:M, i=1:doc_length[m]
    z = rand(Categorical(theta[:,m]))
    w[n] = rand(Categorical(phi[:,z]))
    doc[n] = m
    n = n + 1
end


const ldastandata = [
  Dict(
  "K" => K,
  "V" => V,
  "M" => M,
  "N" => N,
  "w" => w,
  "doc" => doc,
  "alpha" => alpha,
  "beta" => beta,
#   "phi" => phi,
#   "theta" => theta,
#   "D" => M * K + K * V
  )
]

println(ldastandata)

using HDF5, JLD

save(Pkg.dir("Turing")*"/example-models/stan-models/ldastandata.jld", "data", ldastandata)
