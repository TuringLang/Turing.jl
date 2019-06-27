using Distributions: MvNormal, logpdf
using ForwardDiff: gradient
using AdvancedHMC

# Define the target distribution and its gradient
const D = 10
const target = MvNormal(zeros(D), ones(D))
logπ(θ::AbstractVector{<:Real}) = logpdf(target, θ)
∂logπ∂θ(θ::AbstractVector{<:Real}) = gradient(logπ, θ)

# Sampling parameter settings
n_samples = 100_000
n_adapts = 2_000

# Initial points
θ_init = randn(D)

# Define metric space, Hamiltonian and sampling method
metric = DenseEuclideanMetric(D)
h = Hamiltonian(metric, logπ, ∂logπ∂θ)
prop = NUTS(Leapfrog(find_good_eps(h, θ_init)))
adaptor = StanHMCAdaptor(
    n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, prop.integrator.ϵ))

# Sampling
# samples = sample(h, prop, θ_init, n_samples, adaptor, n_adapts)
bench_res = @tbenchmark_expr("NUTS(Leapfrog(...))",
                             sample(h, prop, θ_init, n_samples, adaptor, n_adapts))

# bench_res[4].names = ["phi[1]", "phi[2]", "phi[3]", "phi[4]"]
logd = build_logd("MvNormal-Benchmark", bench_res...)

print_log(logd)
send_log(logd)
