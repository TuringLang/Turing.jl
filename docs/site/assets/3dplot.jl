using Plots
using StatsPlots
using Turing
using Bijectors
using Random

Random.seed!(0)

# Define a strange model.
@model gdemo(x, y) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))
    m = m*sin(m)+s*cos(s)
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
    return s, m
end

# Define our data points.
x = 1.5
y = 2

# Set up the model call, sample from the prior.
model = gdemo(x, y)
vi = Turing.VarInfo()
model(vi, Turing.SampleFromPrior())
vi.flags["trans"] = [true, false]

# Define a function to optimize.
function evaluate(m1, m2)
    vi.vals .= [m1, m2]
    model(vi, Turing.SampleFromPrior())
    -vi.logp
end

function plot_sampler(chain)
    # Extract values from chain.
    ss = link.(Ref(InverseGamma(2, 3)), chain[:s])
    ms = chain[:m]
    lps = chain[:lp]

    # How many surface points to sample.
    granularity = 500

    # Range start/stop points.
    spread = 0.5
    σ_start = minimum(ss) - spread * std(ss); σ_stop = maximum(ss) + spread * std(ss);
    μ_start = minimum(ms) - spread * std(ms); μ_stop = maximum(ms) + spread * std(ms);
    σ_rng = collect(range(σ_start, stop=σ_stop, length=granularity))
    μ_rng = collect(range(μ_start, stop=μ_stop, length=granularity))

    # Make surface plot.
    p = surface(σ_rng, μ_rng, evaluate,
        camera=(25, 65),
        ticks=nothing,
        colorbar=false,
        color=:inferno)

    line_range = 1:length(ms)

    # Add sampler plot.
    # scatter3d!(p, ss[line_range], ms[line_range], -lps[line_range],
    #     color =:viridis, zcolor=collect(line_range),
    #     legend=false, colorbar=false,
    #     m=(2, 1, Plots.stroke(0)))

    plot3d!(ss[line_range], ms[line_range], -lps[line_range],
        lc =:viridis, line_z=collect(line_range),
        legend=false, colorbar=false, alpha=0.5)

    # Plots.svg(joinpath(@__DIR__, "sampler")
    return p
end


# Sample
smodel = gdemo(x, y)
# c = sample(smodel, NUTS(1000, 1.0))
c = sample(smodel, MH(1000))
plot_sampler(c)
