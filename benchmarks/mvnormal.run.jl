using Turing, TuringBenchmarks.TuringTools

# Define the target distribution and its gradient
const D = 10

@model target(dim) = begin
   Θ = Vector{Real}(undef, dim)
   θ ~ MvNormal(zeros(D), ones(dim))
end

# Sampling parameter settings
n_samples = 100_000
n_adapts = 2_000

# Sampling
bench_res = @tbenchmark_expr("NUTS(Leapfrog(...))",
                             sample(target(D), HMC(n_samples, 0.1, 5)));

LOG_DATA = build_log_data("MvNormal-Benchmark", bench_res...)
print_log(LOG_DATA)
