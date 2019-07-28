using Turing

prior = Beta(2,2)
obs = [0,1,0,1,1,1,1,1,1,1]
exact = Beta(prior.α + sum(obs), prior.β + length(obs) - sum(obs))
meanp = exact.α / (exact.α + exact.β)

@model testbb(obs) = begin
    p ~ Beta(2,2)
    x ~ Bernoulli(p)
    for i = 1:length(obs)
        obs[i] ~ Bernoulli(p)
    end
    p, x
end

gibbs = Gibbs(HMC(0.2, 3, :p), PG(10, :x))
chn_g = sample(testbb(obs), gibbs, 1500) ############ not linked somewhere XXX:
