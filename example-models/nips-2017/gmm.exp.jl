
include("gmm.helper.jl")
include("gmm.model.jl")

N = 100000

particle_nums = [5, 10, 50, 100]
nrows = length(particle_nums)
HMC_runs = [10, 100, 500, 1000] .+ 1
ncols = length(HMC_runs)

vs_all = []

for K = HMC_runs

var_pn_plots = []

for pn = particle_nums

println("Running Gibbs")
# chain_gibbs = sample(gmm_gen(p, μ, σ), Gibbs(round(Int,N/K), PG(10, 1, :z), NUTS(K-1, 500, 0.65, :x); thin=false))
chain_gibbs = sample(gmm_gen(p, μ, σ), Gibbs(round(Int,N/K), PG(10, 1, :z), HMC(K-1, 0.2, 4, :x); thin=false))
#chain_gibbs = load(TPATH*"/nips-2017/gmm/gibbs-chain-2.5.jld")["chain"]
#save(TPATH*"/nips-2017/gmm/gibbs-chain-0.5.jld", "chain", chain_gibbs)
x_gibbs = map(x_arr -> x_arr[1], chain_gibbs[:x])

println("Running NUTS")
chain_nuts = sample(gmm_gen_marg(p, μ, σ), NUTS(N, 0.65))
#chain_nuts = load(TPATH*"/nips-2017/gmm/nuts-chain-2.5.jld")["chain"]
#save(TPATH*"/nips-2017/gmm/nuts-chain-0.5.jld", "chain", chain_nuts)
x_nuts = map(x_arr -> x_arr[1], chain_nuts[:x])

gibbs_layer = layer(x=x_gibbs, Geom.density, Theme(default_color=colors[1]))
nuts_layer = layer(x=x_nuts, Geom.density, Theme(default_color=colors[2]))
contour_layer = layer([make_norm_pdf(p, μ, σ)], -5, 20, Theme(default_color=colors[3]))

layers = [gibbs_layer, nuts_layer, contour_layer]
labels = ["Gibbs", "NUTS", "Exact"]

select = [1,2,3]
gmm_density = plot(layers..., Guide.manual_color_key("", labels, colors),
                   Coord.cartesian(xmin=-5, xmax=20, ymin=0, ymax=1.0),
                   Guide.xlabel(nothing), Guide.ylabel("Density"), Guide.title("PG($pn,1)+NUTS($K,500,0.64)"))

push!(var_pn_plots, gmm_density)

end

vs = vstack(var_pn_plots...)
push!(vs_all, vs)

end

s_all = hstack(vs_all...)

Gadfly.push_theme(:default)

draw(PDF(TPATH*"/example-models/nips-2017/gmm-density-vary-np.pdf", (4*ncols)inch, (2*nrows)inch), s_all)
