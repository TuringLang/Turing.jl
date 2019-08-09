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

LOG_DATA = @tbenchmark(HMC(1000, 1.5, 3), constrained_test, data)

log_report("Dummy benchmark finished!")

print_log(LOG_DATA)

log_report("Dummy benchmark printed!")
