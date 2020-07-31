# TODO: lots of imports will have been done in runtests.jl hence irrelevant
using Zygote, ReverseDiff, Memoization, Turing; turnprogress(false)
using Turing: AIS, Sampler
using Pkg
using Random
using Test
using DynamicPPL: getlogp, setlogp!, SampleFromPrior, PriorContext, VarInfo
using Plots
using AdvancedMH

# TODO: replace plots by some other testing metric

# TODO: find another model that's less trivial but still conjugate

# declare models

@model model_1(x) = begin
    # latent
    z = Vector{Real}(undef, 2)
    z[1] ~ Normal()
    z[2] ~ Normal()
    # observed
    x ~ Normal(z[1], z[2] * z[2])
end
model_1 = model_1(1.)

@model model_2(x) = begin
    # latent
    inv_theta ~ Gamma(2,3)
    theta = 1/inv_theta
    # observed
    x ~ Weibull(1,theta) 
end
model_2 = model_2(5.)

# declare algorithm and sampler

schedule = 0.1:0.1:0.9
proposal_kernels = [AdvancedMH.RWMH(Normal()) for i in 1:9]
alg = AIS(proposal_kernels, schedule)
spl = Sampler(alg, model_1)

# TODO: test sample_init!

# test prior_step(model)

list_samples = []
for i in 1:50
    append!(list_samples, prior_step(model)[1]) # prior_step(model) returns an array
end
p = histogram(list_samples)
png(p, "/Users/js/prior_samples_hist.png")

# TODO: test intermediate_step(j, spl, current_state, accum_logweight)

# test gen_logjoint

logjoint = gen_logjoint(spl.state.vi, model, spl)
logjoint_values = logjoint.(list_args)
p = plot(interval, logjoint_values)

# test gen_logprior

logprior = gen_logprior(spl.state.vi, model, spl)
logprior_values = logprior.(list_args)
plot!(p, interval, logprior_values)

# test gen_log_unnorm_tempered

for beta in 0.1:0.1:0.9
    log_unnorm_tempered = gen_log_unnorm_tempered(logprior, logjoint, beta)
    log_unnorm_tempered_values = log_unnorm_tempered.(list_args)
    plot!(p, interval, log_unnorm_tempered_values)
end

png(p, "/Users/js/prior_and_joint.png")

# TODO: test sample_end!
