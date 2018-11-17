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
    n, M = ve.n, ve.M
    @assert n >= 2 "Cannot get variance with only one sample"
    return (n / ((n + 5) * (n - 1))) .* M .+ 1e-3 * (5.0 / (n + 5))
end

# TODO: to implement cov estimater
mutable struct CovarEstimator{TI<:Integer,TF<:Real}
end

######################
### Mutable states ###
######################

mutable struct DPCState{T<:Real} <: AbstractState
    std :: Vector{T}
end

# TODO: to implement full pc
mutable struct FPCState{T<:Real} <: AbstractState
end

################
### Adapters ###
################

abstract type PreConditioner <: AbstractAdapt end

struct NullPC <: PreConditioner end

function getstd(::NullPC)
    return [1.0]
end

struct DiagonalPC{TI<:Integer,TF<:Real} <: PreConditioner
    ve    :: VarEstimator{TI,TF}
    state :: DPCState{TF}
end

function DiagonalPC(d::Integer)
    ve = VarEstimator(0, zeros(d), zeros(d))
    return DiagonalPC(ve, DPCState(ones(d)))
end

function getstd(dpc::DiagonalPC)
    return dpc.state.std
end

struct FullPC{TI<:Integer,TF<:Real} <: PreConditioner
    ce    :: CovarEstimator{TI,TF}
    state :: FPCState{TF}
end
