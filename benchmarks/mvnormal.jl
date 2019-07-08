##
## MvNormal
##

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

##
## MvNormal: ForwardDiff vs BackwardDiff (Tracker)
##

using Turing, LinearAlgebra
using BenchmarkTools

@model mdemo(d, N) = begin
    Θ = Vector(undef, N)
   for n=1:N
      Θ[n] ~ d
   end
end

dim2 = 250
A   = rand(Wishart(dim2, Matrix{Float64}(I, dim2, dim2)))
d   = MvNormal(zeros(dim2), A)

# ForwardDiff
Turing.setadbackend(:forward_diff)
@benchmark chain = sample(mdemo(d, 1), HMC(5000, 0.1, 5))

#BackwardDiff
Turing.setadbackend(:reverse_diff)
@benchmark chain = sample(mdemo(d, 1), HMC(5000, 0.1, 5))

# build log and send data back to github.
