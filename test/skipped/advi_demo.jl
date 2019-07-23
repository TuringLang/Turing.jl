using Revise

using Random

using Turing
using Turing: Variational

# setup for plotting
using Plots, StatsPlots, LaTeXStrings
pyplot()

# used to compute closed form expression of posterior
using ConjugatePriors

# define Turing model
@model model(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))  # `Normal(μ, σ)` has mean μ and variance σ², i.e. parametrize with std. not variance
    for i = 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

const seeds = [125, 245, 1]
const ad_modes = [:forward_diff, :reverse_diff]

for seed ∈ seeds
    @info seed
    
    for ad_mode ∈ ad_modes
        @info ad_mode
        setadbackend(ad_mode)
        
        # set random seed
        Random.seed!(seed)

        # generate data
        x = randn(1, 2000);

        # construct model
        m = model(x)
        
        # ADVI
        opt = Variational.TruncatedADAGrad() # optimizer
        advi = ADVI(10, 100)                 # <: VariationalInference
        q = vi(m, advi; optimizer = opt)     # => MeanField <: VariationalPosterior
        
        elbo = Variational.ELBO()            # <: VariationalObjective

        θ = vcat(q.μ, q.ω)
        # θ = zeros(2 * length(q))

        history = [elbo(advi, q, m, 1000)]     # history of objective evaluations

        # construct animation
        anim = @animate for j = 1:25
            # global q
            Variational.optimize!(elbo, advi, q, m, θ; optimizer = opt)
            μ, ω = θ[1:length(q)], θ[length(q) + 1:end]
            
            q = Variational.MeanField(μ, ω, q.dists, q.ranges)
            samples = rand(q, 2000)

            # quick check
            println([mean(samples, dims=2), [var(x), mean(x)]])

            # plotting code assumes (samples, dim) shape so we just transpose
            samples = transpose(samples)

            # closed form computation
            # notation mapping has been verified by explicitly computing expressions
            # in "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy
            μ₀ = 0.0 # => μ
            κ₀ = 1.0 # => ν, which scales the precision of the Normal
            α₀ = 2.0 # => "shape"
            β₀ = 3.0 # => "rate", which is 1 / θ, where θ is "scale"

            # prior
            pri = NormalGamma(μ₀, κ₀, α₀, β₀)

            # posterior
            post = posterior(pri, Normal, x)

            # marginal distribution of τ = 1 / σ²
            # Eq. (90) in "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy
            # `scale(post)` = θ
            p_τ = Gamma(post.shape, scale(post))
            p_σ²_pdf = z -> pdf(p_τ, 1 / z) # τ => 1 / σ² 

            # marginal of μ
            # Eq. (91) in "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy
            p_μ = TDist(2 * post.shape)

            μₙ = post.mu    # μ → μ
            κₙ = post.nu    # κ → ν
            αₙ = post.shape # α → shape
            βₙ = post.rate  # β → rate

            # numerically more stable but doesn't seem to have effect; issue is probably internal to
            # `pdf` which needs to compute ≈ Γ(1000) 
            p_μ_pdf = z -> exp(logpdf(p_μ, (z - μₙ) * exp(- 0.5 * log(βₙ) + 0.5 * log(αₙ) + 0.5 * log(κₙ))))
            # p_μ_pdf1 = z -> pdf(p_μ, (z - μₙ) / √(βₙ / (αₙ * κₙ)))

            # posterior plots
            p1 = plot();
            density!(samples[:, 1], label = "s (ADVI)", color = :blue, linestyle = :dash);
            histogram!(samples[:, 1], label = "", normed = true, alpha = 0.3, color = :blue);

            # normalize using Riemann approx. because of (almost certainly) numerical issues
            Δ = 0.001
            r = 0.75:0.001:1.50
            norm_const = sum(p_σ²_pdf.(r) .* Δ)
            plot!(r, p_σ²_pdf, label = "s (posterior)", color = :red);
            vline!([var(x)], label = "s (data)", linewidth = 1.5, color = :black, alpha = 0.7);
            xlims!(0.5, 1.5);
            title!("$(j * advi.max_iters) steps");

            p2 = plot();
            density!(samples[:, 2], label = "m (ADVI)", color = :blue, linestyle = :dash);
            histogram!(samples[:, 2], label = "", normed = true, alpha = 0.3, color = :blue);


            # normalize using Riemann approx. because of (almost certainly) numerical issues
            Δ = 0.0001
            r = -0.1 + mean(x):Δ:0.1 + mean(x)
            norm_const = sum(p_μ_pdf.(r) .* Δ)
            plot!(r, z -> p_μ_pdf(z) / norm_const, label = "m (posterior)", color = :red);
            vline!([mean(x)], label = "m (data)", linewidth = 1.5, color = :black, alpha = 0.7);

            xlims!(-0.25, 0.25);

            # visualize evolution of objective wrt. optimization iterations
            obj = elbo(advi, q, m, 1000)
            @info "ELBO" obj
            push!(history, obj)
            p3 = plot();
            plot!(1:advi.max_iters:length(history) * advi.max_iters, history, label = "")
            title!("ELBO = $obj")

            # plot the latest 25 objective evaluations to visualize trend
            p4 = plot();
            plot!(history[max(1, end - 10):end], label = "")

            p = plot(p1, p2, p3, p4; layout = (4, 1))
            
            @info "[$j] Done"
            p
        end
        gif(anim, "advi_w_elbo_fps15_$(seed)_$(ad_mode).gif", fps = 15)
    end
end
