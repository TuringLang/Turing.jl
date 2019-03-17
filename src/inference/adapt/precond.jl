##########################
### Variance estimator ###
##########################
abstract type VarEstimator{TI,TF} end

# Ref： https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_var_estimator.hpp
mutable struct WelfordVar{TI, TF} <: VarEstimator{TI, TF}
    n :: TI
    μ :: Vector{TF}
    M :: Vector{TF}
end

function reset!(wv::WelfordVar{TI, TF}) where {TI<:Integer, TF<:Real}
    wv.n = zero(TI)
    wv.μ .= zero(TF)
    wv.M .= zero(TF)
end

function add_sample!(wv::WelfordVar, s::AbstractVector)
    wv.n += 1
    δ = s .- wv.μ
    wv.μ .+= δ ./ wv.n
    wv.M .+= δ .* (s .- wv.μ)
end

# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/var_adaptation.hpp
function get_var(wv::VarEstimator{TI,TF})::Vector{TF} where {TI<:Integer,TF<:Real}
    n, M = wv.n, wv.M
    @assert n >= 2 "Cannot get variance with only one sample"
    return (n / ((n + 5) * (n - 1))) .* M .+ 1e-3 * (5 / (n + 5))
end

abstract type CovarEstimator{TI<:Integer,TF<:Real} end

# NOTE: this naive covariance estimator is used only in testing
mutable struct NaiveCovar{TI,TF} <: CovarEstimator{TI,TF}
    n :: TI
    S :: Vector{Vector{TF}}
end

function add_sample!(nc::NaiveCovar, s::AbstractVector)
    nc.n += 1
    push!(nc.S, s)
end

function reset!(nc::NaiveCovar{TI,TF}) where {TI<:Integer,TF<:Real}
    nc.n = zero(TI)
    nc.S = Vector{Vector{TF}}()
end

function get_covar(nc::NaiveCovar{TI,TF})::Matrix{TF} where {TI<:Integer,TF<:Real}
    @assert nc.n >= 2 "Cannot get variance with only one sample"
    return Statistics.cov(nc.S)
end

# Ref: https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_covar_estimator.hpp
mutable struct WelfordCovar{TI<:Integer,TF<:Real} <: CovarEstimator{TI,TF}
    n :: TI
    μ :: Vector{TF}
    M :: Matrix{TF}
end

function reset!(wc::WelfordCovar{TI,TF}) where {TI<:Integer,TF<:Real}
    wc.n = zero(TI)
    wc.μ .= zero(TF)
    wc.M .= zero(TF)
end

function add_sample!(wc::WelfordCovar, s::AbstractVector)
    wc.n += 1
    δ = s .- wc.μ
    wc.μ .+= δ ./ wc.n
    wc.M .+= (s .- wc.μ) * δ'
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/covar_adaptation.hpp
function get_covar(wc::WelfordCovar{TI,TF})::Matrix{TF} where {TI<:Integer,TF<:Real}
    n, M = wc.n, wc.M
    @assert n >= 2 "Cannot get variance with only one sample"
    return (n / ((n + 5) * (n - 1))) .* M + 1e-3 * (5 / (n + 5)) * LinearAlgebra.I
end


################
### Adapters ###
################

abstract type PreConditioner <: AbstractAdapter end

struct UnitPreConditioner <: PreConditioner end

function Base.string(::UnitPreConditioner)
    return string([1.0])
end

struct DiagPreConditioner{TF<:Real, Tve <: VarEstimator} <: PreConditioner
    ve  :: Tve
    std :: Vector{TF}
end

function DiagPreConditioner(d::Integer)
    ve = WelfordVar(0, zeros(d), zeros(d))
    return DiagPreConditioner(ve, Vector(ones(d)))
end

function Base.string(dpc::DiagPreConditioner)
    return string(dpc.std)
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

struct DensePreConditioner{TF<:Real, Tce <: CovarEstimator} <: PreConditioner
    ce    :: Tce
    covar :: Matrix{TF}
end

function DensePreConditioner(d::Integer)
    ce = WelfordCovar(0, zeros(d), zeros(d,d))
    # TODO: take use of the line below when we have an interface to set which pre-conditioner to use
    # ce = NaiveCovar(0, Vector{Vector{Float64}}())
    return DensePreConditioner(ce, LinearAlgebra.diagm(0 => ones(d)))
end

function Base.string(dpc::DensePreConditioner)
    return string(LinearAlgebra.diag(dpc.covar))
end

function adapt!(dpc::DensePreConditioner, θ, is_addsample::Bool, is_updatestd::Bool)
    if is_addsample
        add_sample!(dpc.ce, θ)
    end
    if is_updatestd
        covar = get_covar(dpc.ce)
        dpc.covar .= covar
        reset!(dpc.ce)
        return true
    end
    return false
end
