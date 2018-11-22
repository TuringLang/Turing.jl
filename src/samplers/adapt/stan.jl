@static if isdefined(Turing, :CmdStan)
  function DualAveraging(spl::Sampler{<:AdaptiveHamiltonian}, adapt_conf::CmdStan.Adapt, ϵ::Real)
      # Hyper parameters for dual averaging
      γ = adapt_conf.gamma
      t_0 = adapt_conf.t0
      κ = adapt_conf.kappa
      δ = adapt_conf.delta
      return DualAveraging(γ, t_0, κ, δ, DAState(ϵ))
  end
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
