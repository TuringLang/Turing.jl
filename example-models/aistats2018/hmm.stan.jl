include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
using Stan, HDF5, JLD

const hmmstanmodel = "
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

const hmm_semisup_data = load(Pkg.dir("Turing")*"/example-models/nips-2017/hmm_semisup_data.jld")["data"]

hmmstan = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.025),Stan.diag_e(),0.005,0.0), save_warmup=true,adapt=Stan.Adapt(engaged=false)), num_samples=1000, num_warmup=0, thin=1, name="Hidden_Markov", model=hmmstanmodel, nchains=1);

rc, hmm_stan_sim = stan(hmmstan, hmm_semisup_data, CmdStanDir=CMDSTAN_HOME, summary=false);

sv_time = get_stan_time("Hidden_Markov")
println("Time used:", sv_time)