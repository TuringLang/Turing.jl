immutable eNUTS <: InferenceAlgorithm
  n_samples ::  Int       # number of samples
  step_size ::  Float64   # leapfrog step size
  space     ::  Set       # sampling space, emtpy means all
  group_id  ::  Int

  eNUTS(step_size::Float64) = new(1, step_size, Set(), 0)
  eNUTS(n_samples::Int, step_size::Float64) = new(n_samples, step_size, Set(), 0)
  eNUTS(n_samples::Int, step_size::Float64, space...) = new(n_samples, step_size, isa(space, Symbol) ? Set([space]) : Set(space), 0)
  eNUTS(alg::eNUTS, new_group_id::Int) = new(alg.n_samples, alg.step_size, alg.space, new_group_id)
end

function step(model, spl::Sampler{eNUTS}, vi::VarInfo, is_first::Bool)
  if is_first
    true, vi
  else
    ϵ = spl.alg.step_size

    dprintln(2, "sampling momentum...")
    p = sample_momentum(vi, spl)

    dprintln(3, "X -> R...")
    vi = link(vi, spl)

    dprintln(3, "sample slice variable u (in log scale)")
    logu = log(rand()) + (-find_H(p, model, vi, spl))


    θm, θp, rm, rp, j, vi_new, n, s = deepcopy(vi), deepcopy(vi), deepcopy(p), deepcopy(p), 0, deepcopy(vi), 1, 1

    while s == 1
      v_j = rand([-1, 1]) # Note: this variable actually does not depend on j;
                          #       it is set as `v_j` just to be consistent to the paper
      if v_j == -1
        θm, rm, _, _, θ′, n′, s′, reject = build_tree(θm, rm, logu, v_j, j, ϵ, model, spl)
      else
        _, _, θp, rp, θ′, n′, s′, reject = build_tree(θp, rp, logu, v_j, j, ϵ, model, spl)
      end

      if ~reject && s′ == 1 && rand() < min(1, n′ / n)
        vi_new = deepcopy(θ′)
      end
      n = n + n′
      
      s = s′ * (direction(θm, θp, rm, model, spl) >= 0 ? 1 : 0) * (direction(θm, θp, rp, model, spl) >= 0 ? 1 : 0)
      j = j + 1
    end

    dprintln(3, "R -> X...")
    vi_new = invlink(vi_new, spl)

    cleandual!(vi_new)

    true, vi_new
  end
end

function build_tree(θ, r, logu, v, j, ϵ, model, spl)
  doc"""
    - θ     : model parameter
    - r     : momentum variable
    - logu  : slice variable (in log scale)
    - v     : direction ∈ {-1, 1}
    - j     : depth
    - ϵ     : leapfrog step size
  """
  if j == 0
    # Base case - take one leapfrog step in the direction v.
    θ′, r′, reject = leapfrog(θ, r, 1, v * ϵ, model, spl)
    n′ = reject ?
         0 :
         (logu <= -find_H(r′, model, θ′, spl)) ? 1 : 0
    s′ = reject ?
         0 :
         (logu < Δ_max - find_H(r′, model, θ′, spl)) ? 1 : 0
    return deepcopy(θ′), deepcopy(r′), deepcopy(θ′), deepcopy(r′), deepcopy(θ′), n′, s′, reject
  else
    # Recursion - build the left and right subtrees.
    θm, rm, θp, rp, θ′, n′, s′, reject = build_tree(θ, r, logu, v, j - 1, ϵ, model, spl)
    if reject
      return θm, rm, θp, rp, θ′, n′, s′, true
    end
    if s′ == 1
      if v == -1
        θm, rm, _, _, θ′′, n′′, s′′, reject = build_tree(θm, rm, logu, v, j - 1, ϵ, model, spl)
      else
        _, _, θp, rp, θ′′, n′′, s′′, reject = build_tree(θp, rp, logu, v, j - 1, ϵ, model, spl)
      end
      if ~reject && rand() < n′′ / (n′ + n′′)
        θ′ = deepcopy(θ′′)
      end
      s′ = s′′ * (direction(θm, θp, rm, model, spl) >= 0 ? 1 : 0) * (direction(θm, θp, rp, model, spl) >= 0 ? 1 : 0)
      n′ = n′ + n′′
    end
    return θm, rm, θp, rp, θ′, n′, s′, reject
  end
end
