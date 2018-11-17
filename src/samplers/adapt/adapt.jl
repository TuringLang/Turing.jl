include("precond.jl")
include("stepsize.jl")

######################
### Mutable states ###
######################

mutable struct TPState{T<:Integer} <: AbstractState
    window_size :: T
    next_window :: T
end

################
### Adapters ###
################

abstract type WindowAdapt <: AbstractAdapt end

struct ThreePhase{T<:Integer} <: WindowAdapt
  init_buffer :: T
  term_buffer :: T
  state       :: TPState{T}
end

function ThreePhase(init_buffer, term_buffer, window_size)
    next_window = init_buffer + window_size - 1
    return ThreePhase(init_buffer, term_buffer, TPState(window_size, next_window))
end

@static if isdefined(Turing, :CmdStan)

  function ThreePhase(adapt_conf::CmdStan.Adapt)
      # Three phases settings
      init_buffer = adapt_conf.init_buffer
      term_buffer = adapt_conf.term_buffer
      window_size = adapt_conf.window
      return ThreePhase(init_buffer, term_buffer, window_size)
  end

end

struct NullWindow <: WindowAdapt end

struct CompositeAdapt{T<:AbstractAdapt} <: AbstractAdapt
  adapts :: Vector{T}
end

# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.

mutable struct WarmUpManager{T<:Integer}
    n        :: T
    n_adapts :: T
    pc       :: PreConditioner
    ssa      :: StepSizeAdapt
    win      :: WindowAdapt
end

function WarmUpManager(model::Function, spl::Sampler{<:AdaptiveHamiltonian}, vi::VarInfo)
    # Diagonal pre-conditioner
    pc = DiagonalPC(length(vi[spl]))

    # Initialize by Stan if Stan is installed
    @static if isdefined(Turing, :CmdStan)
        # CmdStan.Adapt
        ssa = DualAveraging(spl.info[:adapt_conf])
        win = ThreePhase(spl.info[:adapt_conf])
    else
        # If wum is not initialised by Stan (when Stan is not avaible),
        # initialise wum by common default values.
        ssa = DualAveraging(0.05, 10.0, 0.75, spl.alg.delta, DAState(model, spl, vi))
        win = ThreePhase(75, 50, 25)
    end

    return WarmUpManager(1, spl.alg.n_adapts, pc, ssa, win)
end

function WarmUpManager(model::Function, spl::Sampler{<:StaticHamiltonian}, vi::VarInfo)
    fss = FixedStepSize(spl.alg.epsilon)
    return WarmUpManager(1, 0, NullPC(), fss, NullWindow())
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function in_adaptsation(wum::WarmUpManager)
    tp = wum.win
    return (wum.n >= tp.init_buffer) &&
        (wum.n < wum.n_adapts - tp.term_buffer) &&
        (wum.n != wum.n_adapts)
end

function is_window_end(wum::WarmUpManager)
    tp = wum.win
    return (wum.n == tp.state.next_window) && (wum.n != wum.n_adapts)
end

function compute_next_window(wum::WarmUpManager)
    tp = wum.win
    if ~(tp.state.next_window == wum.n_adapts - tp.term_buffer - 1)

        tp.state.window_size *= 2
        tp.state.next_window = wum.n + tp.state.window_size

        if ~(tp.state.next_window == wum.n_adapts - tp.term_buffer - 1)
            next_window_boundary = tp.state.next_window + 2 * tp.state.window_size

            if (next_window_boundary >= wum.n_adapts - tp.term_buffer)
                tp.state.next_window = wum.n_adapts - tp.term_buffer - 1
            end
        end
    end
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
function update_precond!(wum::WarmUpManager, θ::AbstractVector{<:Real})
    dpc = wum.pc
    if in_adaptsation(wum)
        add_sample!(dpc.ve, θ)
    end
    if is_window_end(wum)
        var = get_var(dpc.ve)
        dpc.state.std = sqrt.(var)
        reset!(dpc.ve)
        return true
    end
    return false
end

function update_stepsize!(wum::WarmUpManager, stats::Real)
    da = wum.ssa
    adapt_stepsize!(da, stats)
    if is_window_end(wum) || wum.n == wum.n_adapts
        ϵ = exp(da.state.x_bar)
        da.state.ϵ = ϵ
        da.state.μ = computeμ(ϵ)
        reset!(da.state)
    end
    if wum.n == wum.n_adapts
        ϵ = exp(da.state.x_bar)
        da.state.ϵ = ϵ
        @info " Adapted ϵ = $ϵ, $(wum.n_adapts) iterations is used for adaption."
    end
end

function adapt!(wum::WarmUpManager, stats::Real, θ; adapt_ϵ=false, adapt_M=false)
    if wum.n <= wum.n_adapts
        if adapt_ϵ
            update_stepsize!(wum, stats)
        end

        if adapt_M
            update_precond!(wum, θ)
        end

        # If window ends, compute next window
        is_window_end(wum) && compute_next_window(wum)

        wum.n += 1
    end
end
