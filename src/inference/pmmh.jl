"""
    PMMH(n_iters::Int, smc_alg:::SMC, parameters_algs::Tuple{MH})

Particle independant Metropolis–Hastings and
Particle marginal Metropolis–Hastings samplers.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
alg = PMMH(100, SMC(20, :v1), MH(1,:v2))
alg = PMMH(100, SMC(20, :v1), MH(1,(:v2, (x) -> Normal(x, 1))))
```

Arguments:

- `n_iters::Int` : Number of iterations to run.
- `smc_alg:::SMC` : An [`SMC`](@ref) algorithm to use.
- `parameters_algs::Tuple{MH}` : An [`MH`](@ref) algorithm, which includes a
sample space specification.
"""
mutable struct PMMH{T, A<:Tuple} <: InferenceAlgorithm
    n_iters               ::    Int               # number of iterations
    algs                  ::    A                 # Proposals for state & parameters
    space                 ::    Set{T}            # sampling space, emtpy means all
end
function PMMH(n_iters::Int, smc_alg::SMC, parameter_algs...)
    return PMMH(n_iters, tuple(parameter_algs..., smc_alg), Set())
end

PIMH(n_iters::Int, smc_alg::SMC) = PMMH(n_iters, tuple(smc_alg), Set())

function Sampler(alg::PMMH, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    spl = Sampler(alg, info, s)

    alg_str = "PMMH"
    n_samplers = length(alg.algs)
    samplers = Array{Sampler}(undef, n_samplers)

    space = Set{Symbol}()

    for i in 1:n_samplers
        sub_alg = alg.algs[i]
        if isa(sub_alg, Union{SMC, MH})
            samplers[i] = Sampler(sub_alg, model, Selector(Symbol(typeof(sub_alg))))
        else
            error("[$alg_str] unsupport base sampling algorithm $alg")
        end
        if typeof(sub_alg) == MH && sub_alg.n_iters != 1
            warn("[$alg_str] number of iterations greater than 1 is useless for MH since it is only used for its proposal")
        end
        space = union(space, sub_alg.space)
    end

    # Sanity check for space
    if !isempty(space)
        @assert issubset(Set(get_pvars(model)), space) "[$alg_str] symbols specified to samplers ($space) doesn't cover the model parameters ($(Set(get_pvars(model))))"

        if Set(get_pvars(model)) != space
            warn("[$alg_str] extra parameters specified by samplers don't exist in model: $(setdiff(space, Set(get_pvars(model))))")
        end
    end

    info[:old_likelihood_estimate] = -Inf # Force to accept first proposal
    info[:old_prior_prob] = 0.0
    info[:samplers] = samplers

    return spl
end

function step(model, spl::Sampler{<:PMMH}, vi::VarInfo, is_first::Bool)
    violating_support = false
    proposal_ratio = 0.0
    new_prior_prob = 0.0
    new_likelihood_estimate = 0.0
    old_θ = copy(vi[spl])

    Turing.DEBUG && @debug "Propose new parameters from proposals..."
    for local_spl in spl.info[:samplers][1:end-1]
        Turing.DEBUG && @debug "$(typeof(local_spl)) proposing $(local_spl.alg.space)..."
        propose(model, local_spl, vi)
        if local_spl.info[:violating_support] violating_support=true; break end
        new_prior_prob += local_spl.info[:prior_prob]
        proposal_ratio += local_spl.info[:proposal_ratio]
    end

    if !violating_support # do not run SMC if going to refuse anyway
        Turing.DEBUG && @debug "Propose new state with SMC..."
        vi, _ = step(model, spl.info[:samplers][end], vi)
        new_likelihood_estimate = spl.info[:samplers][end].info[:logevidence][end]

        Turing.DEBUG && @debug "computing accept rate α..."
        is_accept, logα = mh_accept(
          -(spl.info[:old_likelihood_estimate] + spl.info[:old_prior_prob]),
          -(new_likelihood_estimate + new_prior_prob),
          proposal_ratio,
        )
    end

    Turing.DEBUG && @debug "decide whether to accept..."
    if !violating_support && is_accept # accepted
        is_accept = true
        spl.info[:old_likelihood_estimate] = new_likelihood_estimate
        spl.info[:old_prior_prob] = new_prior_prob
    else                      # rejected
        is_accept = false
        vi[spl] = old_θ
    end

    return vi, is_accept
end

function sample(  model::Model,
                  alg::PMMH;
                  save_state=false,         # flag for state saving
                  resume_from=nothing,      # chain to continue
                  reuse_spl_n=0             # flag for spl re-using
                )

    spl = Sampler(alg, model)
    if resume_from != nothing
        spl.selector = resume_from.info[:spl].selector
    end
    alg_str = "PMMH"

    # Number of samples to store
    sample_n = spl.alg.n_iters

    # Init samples
    time_total = zero(Float64)
    samples = Array{Sample}(undef, sample_n)
    weight = 1 / sample_n
    for i = 1:sample_n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    # Init parameters
    vi = if resume_from == nothing
        vi_ = VarInfo()
        model(vi_, SampleFromUniform())
        vi_
    else
        resume_from.info[:vi]
    end
    n = spl.alg.n_iters

    # PMMH steps
    accept_his = Bool[]
    PROGRESS[] && (spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0))
    for i = 1:n
      Turing.DEBUG && @debug "$alg_str stepping..."
      time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, i==1)

      if is_accept # accepted => store the new predcits
          samples[i].value = Sample(vi, spl).value
      else         # rejected => store the previous predcits
          samples[i] = samples[i - 1]
      end

      time_total += time_elapsed
      push!(accept_his, is_accept)
      if PROGRESS[]
        haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
      end
    end

    println("[$alg_str] Finished with")
    println("  Running time    = $time_total;")
    accept_rate = sum(accept_his) / n  # calculate the accept rate
    println("  Accept rate         = $accept_rate;")

    if resume_from != nothing   # concat samples
      pushfirst!(samples, resume_from.info[:samples]...)
    end
    c = Chain(-Inf, samples)       # wrap the result by Chain

    if save_state               # save state
      c = save(c, spl, model, vi, samples)
    end

    c
end
