immutable HMCDA <: InferenceAlgorithm
  n_samples ::  Int       # number of samples
  delta     ::  Float64   # target accept rate
  lambda    ::  Int       # target leapfrog length
  space     ::  Set       # sampling space, emtpy means all
  group_id  ::  Int
  HMCDA(delta::Float64, lambda::Int, space...) = HMCDA(1, delta, lambda, space..., 0)
  HMCDA(n_samples::Int, delta::Float64, lambda::Int) = new(n_samples, delta, lambda, Set(), 0)
  HMCDA(n_samples::Int, delta::Float64, lambda::Int, space...) =
    new(n_samples, delta, lambda, isa(space, Symbol) ? Set([space]) : Set(space), 0)
  HMCDA(alg::HMCDA, new_group_id::Int) = new(alg.n_samples, alg.lf_size, alg.lf_num, alg.space, new_group_id)
end

function find_good_eps(vi::VarInfo)
end

function step(model, spl::Sampler{HMCDA}, varInfo::VarInfo, is_first::Bool)
  if is_first
    # Run the model for the first time
    dprintln(2, "initialising...")
    varInfo = runmodel(model, varInfo, spl)
    # Return
    true, varInfo
  else
    # Set parameters
    ϵ, τ = spl.alg.lf_size, spl.alg.lf_num

    dprintln(2, "sampling momentum...")
    p = Dict(uid(k) => randn(length(varInfo[k])) for k in keys(varInfo))
    if ~isempty(spl.alg.space)
      p = filter((k, p) -> getsym(varInfo, k) in spl.alg.space, p)
    end

    dprintln(3, "X -> R...")
    varInfo = link(varInfo, spl)

    dprintln(2, "recording old H...")
    oldH = find_H(p, model, varInfo, spl)

    dprintln(3, "first gradient...")
    val∇E = gradient(varInfo, model, spl)

    dprintln(2, "leapfrog stepping...")
    for t in 1:τ  # do 'leapfrog' for each var
      varInfo, val∇E, p = leapfrog(varInfo, val∇E, p, ϵ, model, spl)
    end

    dprintln(2, "computing new H...")
    H = find_H(p, model, varInfo, spl)

    dprintln(3, "R -> X...")
    varInfo = invlink(varInfo, spl)

    dprintln(2, "computing ΔH...")
    ΔH = H - oldH

    realpart!(varInfo)

    dprintln(2, "decide wether to accept...")
    if ΔH < 0 || rand() < exp(-ΔH)      # accepted
      true, varInfo
    else                                # rejected
      false, varInfo
    end
  end
end
