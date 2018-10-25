abstract type AbstractAdapt end
abstract type StepSizeAdapt <: AbstractAdapt end
abstract type Preconditioner <: AbstractAdapt end

struct DualAveraging <: StepSizeAdapt
end

struct DiagonalPC <: Preconditioner
end

struct FullPC <: Preconditioner
end

struct CompositeAdapt <: AbstractAdapt
  adapts :: Vector{AbstractAdapt}
end


### Old adaptation code below



# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.

# Ref： https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_var_estimator.hpp
mutable struct VarEstimator{T<:Real}
    n::Int
    μ::Vector{T}
    M::Vector{T}
end

function reset!(ve::VarEstimator{T}) where T
    ve.n = 0
    ve.μ .= zero(T)
    ve.M .= zero(T)
    return ve
end

function add_sample!(ve::VarEstimator, s::AbstractVector)
    ve.n += 1
    δ = s .- ve.μ
    ve.μ .+= δ ./ ve.n
    ve.M .+= δ .* (s .- ve.μ)
    return ve
end

# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/var_adaptation.hpp
function get_var(ve::VarEstimator)
    @assert ve.n >= 2
    return (ve.n / ((ve.n + 5) * (ve.n - 1))) .* ve.M .+ 1e-3 * (5.0 / (ve.n + 5))
end



mutable struct WarmUpManager
    adapt_n::Int
    params::Dict{Symbol, Any}
    ve::VarEstimator
end

getindex(wum::WarmUpManager, param) = wum.params[param]

setindex!(wum::WarmUpManager, value, param) = wum.params[param] = value

function init_warm_up_params(vi::VarInfo, spl::Sampler{<:Hamiltonian})
    D = length(vi[spl])
    ve = VarEstimator{Float64}(0, zeros(D), zeros(D))

    wum = WarmUpManager(1, Dict(), ve)

    # Pre-cond
    wum[:stds] = ones(D)

    # Dual averaging
    wum[:ϵ] = [] # why we are using a vector for ϵ
    restart_da(wum)
    wum[:n_warmup] = spl.alg.n_adapt
    wum[:δ] = spl.alg.delta

    # Initialize by Stan if Stan is installed
    @static if isdefined(Turing, :Stan)
        # Stan.Adapt
        adapt_conf = spl.info[:adapt_conf]

        # Hyper parameters for dual averaging
        wum[:γ] = adapt_conf.gamma
        wum[:t_0] = adapt_conf.t0
        wum[:κ] = adapt_conf.kappa

        # Three phases settings
        wum[:init_buffer] = adapt_conf.init_buffer
        wum[:term_buffer] = adapt_conf.term_buffer
        wum[:window_size] = adapt_conf.window
    else
        # If wum is not initialised by Stan (when Stan is not avaible),
         # initialise wum by common default values.
        wum[:γ] = 0.05
        wum[:t_0] = 10.0
        wum[:κ] = 0.75
        wum[:init_buffer] = 75
        wum[:term_buffer] = 50
        wum[:window_size] = 25
    end
    wum[:next_window] = wum[:init_buffer] + wum[:window_size] - 1

    @debug wum.params

    spl.info[:wum] = wum
end

function restart_da(wum::WarmUpManager)
    wum[:m] = 0
    wum[:x_bar] = 0.0
    wum[:H_bar] = 0.0
end

# See NUTS paper sec 3.2.1
function update_da_μ(wum::WarmUpManager, ϵ::Float64)
    wum[:μ] = log(10 * ϵ)
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
function adapt_step_size!(wum::WarmUpManager, stats::Real)

    @debug "adapting step size ϵ..."
    @debug "current α = $(stats)"
    wum[:m] = wum[:m] + 1
    m = wum[:m]

    # Clip average MH acceptance probability.
    stats = stats > 1 ? 1 : stats

    γ = wum[:γ]; t_0 = wum[:t_0]; κ = wum[:κ]; δ = wum[:δ]
    μ = wum[:μ]; x_bar = wum[:x_bar]; H_bar = wum[:H_bar]

    η_H = 1.0 / (m + t_0)
    H_bar = (1.0 - η_H) * H_bar + η_H * (δ - stats)

    x = μ - H_bar * sqrt(m) / γ            # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (1.0 - η_x) * x_bar + η_x * x

    ϵ = exp(x)
    @debug "new ϵ = $(ϵ), old ϵ = $(wum[:ϵ][end])"

    if isnan(ϵ) || isinf(ϵ) || ϵ <= 1e-3
        @warn "Incorrect ϵ = $ϵ; ϵ_previous = $(wum[:ϵ][end]) is used instead."
    else
        push!(wum[:ϵ], ϵ)
        wum[:x_bar], wum[:H_bar] = x_bar, H_bar
    end

    if m == wum[:n_warmup]
        @debug " Adapted ϵ = $ϵ, $m HMC iterations is used for adaption."
    end

end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function in_adaptation(wum::WarmUpManager)
    return (wum.adapt_n >= wum[:init_buffer]) &&
        (wum.adapt_n < wum[:n_warmup] - wum[:term_buffer]) &&
        (wum.adapt_n != wum[:n_warmup])
end

function is_window_end(wum::WarmUpManager)
    return (wum.adapt_n == wum[:next_window]) && (wum.adapt_n != wum[:n_warmup])
end

function compute_next_window(wum::WarmUpManager)

    if ~(wum[:next_window] == wum[:n_warmup] - wum[:term_buffer] - 1)

        wum[:window_size] *= 2
        wum[:next_window] = wum.adapt_n + wum[:window_size]

        if ~(wum[:next_window] == wum[:n_warmup] - wum[:term_buffer] - 1)
            next_window_boundary = wum[:next_window] + 2 * wum[:window_size]

            if (next_window_boundary >= wum[:n_warmup] - wum[:term_buffer])
                wum[:next_window] = wum[:n_warmup] - wum[:term_buffer] - 1
            end
        end
    end
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
function update_pre_cond!(wum::WarmUpManager, θ_new::AbstractVector{<:Real})

  if in_adaptation(wum)
    add_sample!(wum.ve, θ_new)
  end
    if is_window_end(wum)
        compute_next_window(wum)
        var = get_var(wum.ve)
        wum[:stds] = sqrt.(var)
        reset!(wum.ve)
        return true
    end
    return false
end

function adapt!(wum::WarmUpManager, stats::Real, θ_new; adapt_ϵ=false, adapt_M=false)

    if wum.adapt_n < wum[:n_warmup]

        if adapt_ϵ
            adapt_step_size!(wum, stats)
            if is_window_end(wum)
                ϵ = exp(wum[:x_bar])
                push!(wum[:ϵ], ϵ)
                update_da_μ(wum, ϵ)
                restart_da(wum)
            end
        end

        if adapt_M
            update_pre_cond!(wum, θ_new)  # window is updated implicitly.
        else   # update window explicitly.
            is_window_end(wum) && compute_next_window(wum)
        end

        wum.adapt_n += 1

    elseif wum.adapt_n == wum[:n_warmup]

        if adapt_ϵ
            ϵ = exp(wum[:x_bar])
            push!(wum[:ϵ], ϵ)
        end
    end
end
