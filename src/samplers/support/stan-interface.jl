# NOTE:
#   Type fields
#     fieldnames(Stan.Sample)
#       num_samples, num_warmup, save_warmup, thin, adapt, algorithm

#     fieldnames(Stan.Hmc)
#       engine, metric, stepsize, stepsize_jitter

#     fieldnames(Stan.Adapt)
#       engaged, gamma, delta, kappa, t0, init_buffer, term_buffer, window

#   Ref
#     http://goedman.github.io/Stan.jl/latest/index.html#Types-1
sample(mf::T, ss::Stan.Sample) where {T<:Function} = sample(mf, ss.num_samples, ss.num_warmup, ss.save_warmup, ss.thin, ss.adapt, ss.alg)
sample(mf::T, num_samples::Int, num_warmup::Int, save_warmup::Bool, thin::Int, ss::Stan.Sample) where{T<:Function} =
  sample(mf, num_samples, num_warmup, save_warmup, thin, ss.adapt, ss.alg)

sample(mf::T, num_samples::Int, num_warmup::Int, save_warmup::Bool, thin::Int, adapt::Stan.Adapt, alg::Stan.Hmc) where {T<:Function} = begin
  if alg.stepsize_jitter != 0
    error("[Turing.sample] Turing does not support adding noise to stepsize yet.")
  end
  if adapt.engaged == false
    if isa(alg.engine, Stan.Static)   # hmc
      stepnum = Int(round(alg.engine.int_time / alg.stepsize))
      sample(mf, HMC(num_samples, alg.stepsize, stepnum); adapt_conf=adapt)
    elseif isa(alg.engine, Stan.Nuts) # error
      error("[Turing.sample] Stan.Nuts cannot be used with adapt.engaged set as false")
    end
  else
    if isa(alg.engine, Stan.Static)   # hmcda
      sample(mf, HMCDA(num_samples, num_warmup, adapt.delta, alg.engine.int_time); adapt_conf=adapt)
    elseif isa(alg.engine, Stan.Nuts) # nuts
      if alg.metric == Stan.dense_e
        sample(mf, NUTS(num_samples, num_warmup, adapt.delta); adapt_conf=adapt)
      else
        error("[Turing.sample] Turing does not support full covariance matrix for pre-conditioning yet.")
      end
    end
  end
end
