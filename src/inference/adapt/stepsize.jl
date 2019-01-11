######################
### Mutable states ###
######################

mutable struct DAState{TI<:Integer,TF<:Real}
    m     :: TI
    ϵ     :: TF
    μ     :: TF
    x_bar :: TF
    H_bar :: TF
end

function DAState(ϵ::Real)
    μ = computeμ(ϵ) # NOTE: this inital values doesn't affect anything as they will be overwritten
    return DAState(0, ϵ, μ, 0.0, 0.0)
end

function computeμ(ϵ::Real)
    return log(10 * ϵ) # see NUTS paper sec 3.2.1
end

function reset!(dastate::DAState{TI,TF}) where {TI<:Integer,TF<:Real}
    dastate.m = zero(TI)
    dastate.x_bar = zero(TF)
    dastate.H_bar = zero(TF)
end

mutable struct MSSState{T<:Real}
    ϵ :: T
end

################
### Adapters ###
################

abstract type StepSizeAdapter <: AbstractAdapter end

struct FixedStepSize{T<:Real} <: StepSizeAdapter
    ϵ :: T
end

function getss(fss::FixedStepSize)
    return fss.ϵ
end

struct DualAveraging{TI<:Integer,TF<:Real} <: StepSizeAdapter
  γ     :: TF
  t_0   :: TF
  κ     :: TF
  δ     :: TF
  state :: DAState{TI,TF}
end

function DualAveraging(spl::Sampler{<:AdaptiveHamiltonian}, ::Nothing, ϵ::Real)
    return DualAveraging(0.05, 10.0, 0.75, spl.alg.delta, DAState(ϵ))
end

function getss(da::DualAveraging)
    return da.state.ϵ
end

struct ManualSSAdapter{T<:Real} <:StepSizeAdapter
    state :: MSSState{T}
end

function getss(mssa::ManualSSAdapter)
    return mssa.state.ϵ
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
function adapt_stepsize!(da::DualAveraging, stats::Real)
    @debug "adapting step size ϵ..."
    @debug "current α = $(stats)"
    da.state.m += 1
    m = da.state.m

    # Clip average MH acceptance probability.
    stats = stats > 1 ? 1 : stats

    γ = da.γ; t_0 = da.t_0; κ = da.κ; δ = da.δ
    μ = da.state.μ; x_bar = da.state.x_bar; H_bar = da.state.H_bar

    η_H = 1.0 / (m + t_0)
    H_bar = (1.0 - η_H) * H_bar + η_H * (δ - stats)

    x = μ - H_bar * sqrt(m) / γ            # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (1.0 - η_x) * x_bar + η_x * x

    ϵ = exp(x)
    @debug "new ϵ = $(ϵ), old ϵ = $(da.state.ϵ)"

    if isnan(ϵ) || isinf(ϵ)
        @warn "Incorrect ϵ = $ϵ; ϵ_previous = $(da.state.ϵ) is used instead."
    else
        da.state.ϵ = min(one(ϵ), ϵ)
    end
    da.state.x_bar = x_bar
    da.state.H_bar = H_bar
end

function adapt!(da::DualAveraging, stats::Real, is_updateμ::Bool)
    adapt_stepsize!(da, stats)
    if is_updateμ
        da.state.μ = computeμ(da.state.ϵ)
        reset!(da.state)
    end
end
