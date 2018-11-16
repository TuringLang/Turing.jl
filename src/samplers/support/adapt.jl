##########################
### Variance estimator ###
##########################
# Ref： https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_var_estimator.hpp
mutable struct VarEstimator{TI<:Integer,TF<:Real}
    n :: TI
    μ :: Vector{TF}
    M :: Vector{TF}
end

function reset!(ve::VarEstimator{TI,TF}) where {TI<:Integer,TF<:Real}
    ve.n = zero(TI)
    ve.μ .= zero(TF)
    ve.M .= zero(TF)
end

function add_sample!(ve::VarEstimator, s::AbstractVector)
    ve.n += 1
    δ = s .- ve.μ
    ve.μ .+= δ ./ ve.n
    ve.M .+= δ .* (s .- ve.μ)
end

# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/var_adaptation.hpp
function get_var(ve::VarEstimator)
    @assert ve.n >= 2
    return (ve.n / ((ve.n + 5) * (ve.n - 1))) .* ve.M .+ 1e-3 * (5.0 / (ve.n + 5))
end

###################################
### Mutable states for adapters ###
###################################

mutable struct DPCState{T<:Real} <: AbstractState
    std :: Vector{T}
end

mutable struct TPState{T<:Integer} <: AbstractState
    window_size :: T
    next_window :: T
end

mutable struct DAState{TI<:Integer,TF<:Real} <: AbstractState
    m     :: TI
    ϵ     :: TF
    μ     :: TF
    x_bar :: TF
    H_bar :: TF
end

function DAState()
    ϵ = 0.0; μ = 0.0 # NOTE: these inital values doesn't affect anything as they will be overwritten
    return DAState(0, ϵ, μ, 0.0, 0.0)
end

function reset!(dastate::DAState{TI,TF}) where {TI<:Integer,TF<:Real}
    dastate.m = zero(TI)
    dastate.x_bar = zero(TF)
    dastate.H_bar = zero(TF)
end

################
### Adapters ###
################

abstract type AbstractAdapt end
abstract type StepSizeAdapt <: AbstractAdapt end
abstract type Preconditioner <: AbstractAdapt end
abstract type WindowAdapt <: AbstractAdapt end

struct DualAveraging{TI,TF} <: StepSizeAdapt where {TI<:Integer,TF<:Real}
  γ     :: TF
  t_0   :: TF
  κ     :: TF
  δ     :: TF
  state :: DAState{TI,TF}
end

struct ThreePhase{T} <: WindowAdapt where T<:Integer
  init_buffer :: T
  term_buffer :: T
  state       :: TPState{T}
end

function ThreePhase(init_buffer, term_buffer, window_size)
    next_window = init_buffer + window_size - 1
    return ThreePhase(init_buffer, term_buffer, TPState(window_size, next_window))
end

@static if isdefined(Turing, :CmdStan)

  function DualAveraging(adapt_conf::CmdStan.Adapt)
      # Hyper parameters for dual averaging
      γ = adapt_conf.gamma
      t_0 = adapt_conf.t0
      κ = adapt_conf.kappa
      δ = adapt_conf.delta
      return DualAveraging(γ, t_0, κ, δ, DAState())
  end

  function ThreePhase(adapt_conf::CmdStan.Adapt)
      # Three phases settings
      init_buffer = adapt_conf.init_buffer
      term_buffer = adapt_conf.term_buffer
      window_size = adapt_conf.window
      return ThreePhase(init_buffer, term_buffer, window_size)
  end

end

struct DiagonalPC{TI<:Integer,TF<:Real} <: Preconditioner
    ve  :: VarEstimator{TI,TF}
    state :: DPCState{TF}
end

function DiagonalPC(d::Integer)
    ve = VarEstimator(0, zeros(d), zeros(d))
    return DiagonalPC(ve, DPCState(ones(d)))
end

function update!(dpc::DiagonalPC)
    var = get_var(dpc.ve)
    dpc.state.std = sqrt.(var)
end

struct FullPC <: Preconditioner
end

struct CompositeAdapt{T<:AbstractAdapt} <: AbstractAdapt
  adapts :: Vector{T}
end



### Old adaptation code below



# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.

mutable struct WarmUpManager{TI<:Integer,TF<:Real}
    i       :: TI
    n_adapt :: TI
    dpc     :: DiagonalPC{TI,TF}
    da      :: DualAveraging{TI,TF}
    tp      :: ThreePhase{TI}
end

function WarmUpManager(vi::VarInfo, spl::Sampler{<:Hamiltonian})
    # Diagonal pre-conditioner
    dpc = DiagonalPC(length(vi[spl]))

    # Initialize by Stan if Stan is installed
    @static if isdefined(Turing, :CmdStan)
        # CmdStan.Adapt
        da = DualAveraging(spl.info[:adapt_conf])
        tp = ThreePhase(spl.info[:adapt_conf])
    else
        # If wum is not initialised by Stan (when Stan is not avaible),
        # initialise wum by common default values.
        # TODO: remove this fieldnames hack after RFC
        da = if :delta in fieldnames(typeof(spl.alg))
            DualAveraging(0.05, 10.0, 0.75, spl.alg.delta, DAState())
        else
            DualAveraging(0.05, 10.0, 0.75, 0.8, DAState())
        end
        tp = ThreePhase(75, 50, 25)
    end

    # TODO: remove this fieldnames hack after RFC
    n_adapt = :n_adapt in fieldnames(typeof(spl.alg)) ? spl.alg.n_adapt : 0
    return WarmUpManager(1, n_adapt, dpc, da, tp)
end

function update!(dastate::DAState, ϵ::Real)
    dastate.ϵ = ϵ
    dastate.μ = log(10 * ϵ) # see NUTS paper sec 3.2.1
end

function restart_da(wum::WarmUpManager)
    reset!(wum.da.state)
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
function adapt_step_size!(wum::WarmUpManager, stats::Real)

    @debug "adapting step size ϵ..."
    @debug "current α = $(stats)"
    wum.da.state.m = wum.da.state.m + 1
    m = wum.da.state.m

    # Clip average MH acceptance probability.
    stats = stats > 1 ? 1 : stats

    γ = wum.da.γ; t_0 = wum.da.t_0; κ = wum.da.κ; δ = wum.da.δ
    μ = wum.da.state.μ; x_bar = wum.da.state.x_bar; H_bar = wum.da.state.H_bar

    η_H = 1.0 / (m + t_0)
    H_bar = (1.0 - η_H) * H_bar + η_H * (δ - stats)

    x = μ - H_bar * sqrt(m) / γ            # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (1.0 - η_x) * x_bar + η_x * x

    ϵ = exp(x)
    @debug "new ϵ = $(ϵ), old ϵ = $(wum.da.state.ϵ)"

    if isnan(ϵ) || isinf(ϵ) || ϵ <= 1e-3
        @warn "Incorrect ϵ = $ϵ; ϵ_previous = $(wum.da.state.ϵ) is used instead."
    else
        wum.da.state.ϵ = ϵ
        wum.da.state.x_bar, wum.da.state.H_bar = x_bar, H_bar
    end

    if m == wum.n_adapt
        @debug " Adapted ϵ = $ϵ, $m HMC iterations is used for adaption."
    end

end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function in_adaptation(wum::WarmUpManager)
    return (wum.i >= wum.tp.init_buffer) &&
        (wum.i < wum.n_adapt - wum.tp.term_buffer) &&
        (wum.i != wum.n_adapt)
end

function is_window_end(wum::WarmUpManager)
    return (wum.i == wum.tp.state.next_window) && (wum.i != wum.n_adapt)
end

function compute_next_window(wum::WarmUpManager)

    if ~(wum.tp.state.next_window == wum.n_adapt - wum.tp.term_buffer - 1)

        wum.tp.state.window_size *= 2
        wum.tp.state.next_window = wum.i + wum.tp.state.window_size

        if ~(wum.tp.state.next_window == wum.n_adapt - wum.tp.term_buffer - 1)
            next_window_boundary = wum.tp.state.next_window + 2 * wum.tp.state.window_size

            if (next_window_boundary >= wum.n_adapt - wum.tp.term_buffer)
                wum.tp.state.next_window = wum.n_adapt - wum.tp.term_buffer - 1
            end
        end
    end
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
function update_pre_cond!(wum::WarmUpManager, θ_new::AbstractVector{<:Real})

  if in_adaptation(wum)
    add_sample!(wum.dpc.ve, θ_new)
  end
  if is_window_end(wum)
      compute_next_window(wum)
      update!(wum.dpc)
      reset!(wum.dpc.ve)
      return true
  end
  return false
end

function adapt!(wum::WarmUpManager, stats::Real, θ_new; adapt_ϵ=false, adapt_M=false)

    if wum.i < wum.n_adapt

        if adapt_ϵ
            adapt_step_size!(wum, stats)
            if is_window_end(wum)
                ϵ = exp(wum.da.state.x_bar)
                update!(wum.da.state, ϵ)
                restart_da(wum)
            end
        end

        if adapt_M
            update_pre_cond!(wum, θ_new)  # window is updated implicitly.
        else   # update window explicitly.
            is_window_end(wum) && compute_next_window(wum)
        end

        wum.i += 1

    elseif wum.i == wum.n_adapt

        if adapt_ϵ
            ϵ = exp(wum.da.state.x_bar)
            wum.da.state.ϵ = ϵ
        end
    end
end
