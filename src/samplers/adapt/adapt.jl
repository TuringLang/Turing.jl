include("precond.jl")
include("stepsize.jl")

######################
### Mutable states ###
######################

mutable struct TPState{T<:Integer} <: AbstractState
    n           :: T
    window_size :: T
    next_window :: T
end

################
### Adapters ###
################

abstract type CompositeAdapt <: AbstractAdapt end

struct NaiveCompAdapt <: CompositeAdapt
    pc  :: PreConditioner
    ssa :: StepSizeAdapt
end

function getstd(tp::CompositeAdapt)
    return getstd(tp.pc)
end

function getss(tp::CompositeAdapt)
    return getss(tp.ssa)
end

# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.
struct ThreePhase{T<:Integer} <: CompositeAdapt
    n_adapts    :: T
    init_buffer :: T
    term_buffer :: T
    pc          :: PreConditioner
    ssa         :: StepSizeAdapt
    state       :: TPState{T}
end

function get_threephase_params()
    init_buffer, term_buffer, window_size = 75, 50, 25
    next_window = init_buffer + window_size - 1
    return init_buffer, term_buffer, window_size, next_window
end

@static if isdefined(Turing, :CmdStan)
    function get_threephase_params(adapt_conf::CmdStan.Adapt)
        init_buffer = adapt_conf.init_buffer
        term_buffer = adapt_conf.term_buffer
        window_size = adapt_conf.window
        next_window = init_buffer + window_size - 1
        return init_buffer, term_buffer, window_size, next_window
    end
end

function ThreePhase(model::Function, spl::Sampler{<:AdaptiveHamiltonian}, vi::VarInfo)
    # Diagonal pre-conditioner
    pc = DiagonalPC(length(vi[spl]))

    # Initialize by Stan if Stan is installed
    if :adapt_conf in keys(spl.info)
        # CmdStan.Adapt
        ssa = DualAveraging(spl.info[:adapt_conf])
        init_buffer, term_buffer, window_size, next_window = get_threephase_params(spl.info[:adapt_conf])
    else
        # If wum is not initialised by Stan (when Stan is not avaible),
        # initialise wum by common default values.
        ssa = DualAveraging(0.05, 10.0, 0.75, spl.alg.delta, DAState(model, spl, vi))
        init_buffer, term_buffer, window_size, next_window = get_threephase_params()
    end
    tpstate = TPState(0, window_size, next_window)
    return ThreePhase(spl.alg.n_adapts, init_buffer, term_buffer, pc, ssa, tpstate)
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function in_adaptation(tp::ThreePhase)
    return (tp.state.n >= tp.init_buffer) &&
           (tp.state.n < tp.n_adapts - tp.term_buffer) &&
           (tp.state.n != tp.n_adapts)
end

function is_windowend(tp::ThreePhase)
    return (tp.state.n == tp.state.next_window) &&
           (tp.state.n != tp.n_adapts)
end

function compute_next_window!(tp::ThreePhase)
    if ~(tp.state.next_window == tp.n_adapts - tp.term_buffer - 1)
        tp.state.window_size *= 2
        tp.state.next_window = tp.state.n + tp.state.window_size
        if ~(tp.state.next_window == tp.n_adapts - tp.term_buffer - 1)
            next_window_boundary = tp.state.next_window + 2 * tp.state.window_size
            if (next_window_boundary >= tp.n_adapts - tp.term_buffer)
                tp.state.next_window = tp.n_adapts - tp.term_buffer - 1
            end
        end
    end
end

function adapt!(tp::ThreePhase, stats::Real, θ; adapt_ϵ=false, adapt_M=false)
    if tp.state.n < tp.n_adapts
        tp.state.n += 1
        if tp.state.n == tp.n_adapts
            @info " Adapted ϵ = $(getss(tp)), std = $(getstd(tp)); $(tp.state.n) iterations is used for adaption."
        else
            if adapt_ϵ
                is_updateϵ = is_windowend(tp) || tp.state.n == tp.n_adapts
                adapt!(tp.ssa, stats, is_updateϵ)
            end

            # Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
            if adapt_M
                is_addsample, is_updatestd = in_adaptation(tp), is_windowend(tp)
                adapt!(tp.pc, θ, is_addsample, is_updatestd)
            end

            # If window ends, compute next window
            is_windowend(tp) && compute_next_window!(tp)
        end
    end
end
