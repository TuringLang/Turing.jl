using Zygote, ReverseDiff, Memoization, Turing; turnprogress(false)
using Turing: Sampler
using Pkg
using Random
using Test
using DynamicPPL: getlogp, setlogp!, SampleFromPrior, PriorContext, VarInfo
using Plots

# TODO: replace plots by some other testing metric

# TODO: find another model that's less trivial but still conjugate

@model gdemo(x) = begin
    # latent
    z ~ Normal()
    # observed
    x ~ Normal(z, 1.)
end

model = gdemo(1.)

# test sampling from prior

list_samples = []
for i in 1:50
    vi = VarInfo()
    prior_spl = SampleFromPrior() 
    model(vi, prior_spl)
    append!(list_samples, vi[prior_spl][1])
end
p = histogram(list_samples)
png(p, "/Users/js/prior_samples_hist.png")


alg = MH()
spl = Sampler(alg, model)

interval = -3:0.01:3
list_args = [[z] for z in interval]

# test gen_logjoint

logjoint = gen_logjoint(spl.state.vi, model)
logjoint_values = logjoint.(list_args)
p = plot(interval, logjoint_values)

# test gen_logprior

logprior = gen_logprior(spl.state.vi, model)
logprior_values = logprior.(list_args)
plot!(p, interval, logprior_values)

# test gen_log_unnorm_tempered

for beta in 0.1:0.1:0.9
    log_unnorm_tempered = gen_log_unnorm_tempered(logprior, logjoint, beta)
    log_unnorm_tempered_values = log_unnorm_tempered.(list_args)
    plot!(p, interval, log_unnorm_tempered_values)
end

png(p, "/Users/js/prior_and_joint.png")

# TODO: test sample_init!

# TODO: test sample_end!

# TODO: create a function for every intermediate step of the step! function and test it 