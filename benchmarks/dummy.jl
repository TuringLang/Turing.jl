using Turing, BenchmarkHelper

data = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

log_report("Dummy Benchmark started!")

@model constrained_test(obs) = begin
    p ~ Beta(2,2)
    for i = 1:length(obs)
        obs[i] ~ Bernoulli(p)
    end
    p
end

log_report("Dummy model constructed!")

# BENCHMARK_RESULT = @tbenchmark(HMC(1.5, 3), constrained_test, data)
BENCHMARK_RESULT = @tbenchmark_expr "HMC" sample(constrained_test(data),
                                           HMC(1.5, 3),
                                           1000)

log_report("Dummy benchmark finished!")
