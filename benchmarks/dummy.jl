using Turing, ContinuousBenchmarks.TuringTools, ContinuousBenchmarks.Reporter

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

bench_res = @tbenchmark(HMC(1000, 1.5, 3), constrained_test, data)

log_report("Dummy benchmark finished!")

# bench_res[4].names = ["phi[1]", "phi[2]", "phi[3]", "phi[4]"]
LOG_DATA = build_log_data("Dummy-Benchmark", bench_res...)
print_log(LOG_DATA)

log_report("Dummy benchmark reported!")
