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
# https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_covar_estimator.hpp
mutable struct CovarEstimator{TI<:Integer,TF<:Real}
end

# TODO: to implement cov estimater
# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/covar_adaptation.hpp
function get_covar(ve::CovarEstimator)
end
# NOTE: related Hamiltonian change: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/hamiltonians/dense_e_metric.hpp

################
### Adapters ###
################

abstract type PreConditioner <: AbstractAdapter end

struct UnitPreConditioner <: PreConditioner end

function getstd(::UnitPreConditioner)
    return [1.0]
end

struct DiagPreConditioner{TI<:Integer,TF<:Real} <: PreConditioner
    ve  :: VarEstimator{TI,TF}
    std :: Vector{TF}
end

function DiagPreConditioner(d::Integer)
    ve = VarEstimator(0, zeros(d), zeros(d))
    return DiagPreConditioner(ve, Vector(ones(d)))
end

function getstd(dpc::DiagPreConditioner)
    return dpc.std
end

function adapt!(dpc::DiagPreConditioner, θ, is_addsample::Bool, is_updatestd::Bool)
    if is_addsample
        add_sample!(dpc.ve, θ)
    end
    if is_updatestd
        var = get_var(dpc.ve)
        dpc.std .= sqrt.(var)
        reset!(dpc.ve)
        return true
    end
    return false
end

struct DensePreConditioner{TI<:Integer,TF<:Real} <: PreConditioner
    ce  :: CovarEstimator{TI,TF}
    std :: Matrix{TF}
end
