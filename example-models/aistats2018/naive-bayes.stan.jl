using Stan, HDF5, JLD

const nbstanmodel = "
data {
  int C;
  int D;
  int N;
  matrix[D,N] images;
  int<lower=1,upper=C> labels[N];
}
parameters {
  matrix[D,C] mu;
}
model {
  for (c in 1:C)
    for (d in 1:D)
        mu[d,c] ~ normal(0, 10);
      
  for (n in 1:N)
    for (d in 1:D)
        images[d,n] ~ normal(mu[d,labels[n]], 1);
}
"

const nbmnistdata = load(Pkg.dir("Turing")*"/example-models/aistats2018/nbstandata.jld")["data"]

nbstan = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.25),Stan.diag_e(),0.05,0.0), save_warmup=true,adapt=Stan.Adapt(engaged=false)), num_samples=500, num_warmup=0, thin=1, name="Naive_Bayes", model=nbstanmodel, nchains=1);

rc, nb_stan_sim = stan(nbstan, nbmnistdata, CmdStanDir=CMDSTAN_HOME, summary=false);