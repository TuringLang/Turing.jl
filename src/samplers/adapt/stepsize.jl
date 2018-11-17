######################
### Mutable states ###
######################

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

abstract type StepSizeAdapt <: AbstractAdapt end

struct DualAveraging{TI,TF} <: StepSizeAdapt where {TI<:Integer,TF<:Real}
  γ     :: TF
  t_0   :: TF
  κ     :: TF
  δ     :: TF
  state :: DAState{TI,TF}
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

end
