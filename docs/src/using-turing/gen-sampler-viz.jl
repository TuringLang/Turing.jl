ENV["GKS_ENCODING"] = "utf-8" # Allows the use of unicode characters in Plots.jl
using Plots
using StatsPlots
using Turing
using Bijectors
using Random
using DynamicPPL: getlogp, settrans!, getval, reconstruct, vectorize, setval!

# Set a seed.
Random.seed!(0)

# Define a strange model.
@model gdemo(x) = begin
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    bumps = sin(m) + cos(m)
    m = m + 5*bumps
    for i in eachindex(x)
      x[i] ~ Normal(m, sqrt(s²))
    end
    return s², m
end

# Define our data points.
x = [1.5, 2.0, 13.0, 2.1, 0.0]

# Set up the model call, sample from the prior.
model = gdemo(x)
vi = Turing.VarInfo(model)

# Convert the variance parameter to the real line before sampling.
# Note: We only have to do this here because we are being very hands-on.
# Turing will handle all of this for you during normal sampling.
dist = InverseGamma(2,3)
svn = vi.metadata.s.vns[1]
mvn = vi.metadata.m.vns[1]
setval!(vi, vectorize(dist, Bijectors.link(dist, reconstruct(dist, getval(vi, svn)))), svn)
settrans!(vi, true, svn)

# Evaluate surface at coordinates.
function evaluate(m1, m2)
    spl = Turing.SampleFromPrior()
    vi[svn] = [m1]
    vi[mvn] = [m2]
    model(vi, spl)
    getlogp(vi)
end

function plot_sampler(chain; label="")
    # Extract values from chain.
    val = get(chain, [:s, :m, :lp])
    ss = link.(Ref(InverseGamma(2, 3)), val.s)
    ms = val.m
    lps = val.lp

    # How many surface points to sample.
    granularity = 100

    # Range start/stop points.
    spread = 0.5
    σ_start = minimum(ss) - spread * std(ss); σ_stop = maximum(ss) + spread * std(ss);
    μ_start = minimum(ms) - spread * std(ms); μ_stop = maximum(ms) + spread * std(ms);
    σ_rng = collect(range(σ_start, stop=σ_stop, length=granularity))
    μ_rng = collect(range(μ_start, stop=μ_stop, length=granularity))

    # Make surface plot.
    p = surface(σ_rng, μ_rng, evaluate,
          camera=(30, 65),
        #   ticks=nothing,
          colorbar=false,
          color=:inferno,
          title=label)

    line_range = 1:length(ms)

    scatter3d!(ss[line_range], ms[line_range], lps[line_range],
        mc =:viridis, marker_z=collect(line_range), msw=0,
        legend=false, colorbar=false, alpha=0.5,
        xlabel="σ", ylabel="μ", zlabel="Log probability",
        title=label)

    return p
end;

samplers = [
    (Gibbs(HMC(0.01, 5, :s), PG(20, :m)), "Gibbs{HMC, PG}"),
    (HMC(0.01, 10), "HMC"),
    (HMCDA(200, 0.65, 0.3), "HMCDA"),
    (MH(), "MH()"),
    (NUTS(0.65), "NUTS(0.65)"),
    (NUTS(0.95), "NUTS(0.95)"),
    (NUTS(0.2), "NUTS(0.2)"),
    (PG(20), "PG(20)"),
    (PG(50), "PG(50)")]

for (i, (spl, spl_name)) in enumerate(samplers)
    c = sample(model, spl, 1000)
    p = plot_sampler(c, label="$spl_name")
    savefig(joinpath(@__DIR__, "sampler-figs", "samplers-$i.svg"))
end
