using Turing

alg = Gibbs(1000, HMC(1, 0.2, 3, :m), PG(10, 1, :s))
chn = sample(gdemo_default, alg);
