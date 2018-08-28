"""
    gradient(vi::VarInfo, model::Function, spl::Union{Nothing, Sampler})

Function to generate the gradient dictionary, with each prior map to its derivative of the logjoint probibilioty. This function uses chunk-wise forward AD with a chunk of size $(CHUNKSIZE) as default.

Example:

```julia
grad = gradient(vi, model, spl)
end
```
"""
gradient(vi::VarInfo, model::Function) = gradient(vi, model, nothing)
gradient(vi::VarInfo, model::Function, spl::Union{Nothing, Sampler}) = begin

  θ_hash = hash(vi[spl])

  if spl != nothing && haskey(spl.info, :grad_cache)
    if haskey(spl.info[:grad_cache], θ_hash)
      return spl.info[:grad_cache][θ_hash]
    end
  end

  # Initialisation
  grad = Vector{Float64}()

  # Split keys(vi) into chunks,
  @debug "making chunks..."
  vn_chunk = Set{VarName}(); vn_chunks = []; chunk_dim = 0;

  vns = getvns(vi, spl); vn_num = length(vns)

  for i = 1:vn_num
    l = length(getrange(vi, vns[i]))           # dimension for the current variable
    if chunk_dim + l > CHUNKSIZE
      push!(vn_chunks,        # store the previous chunk
            (vn_chunk, chunk_dim))
      vn_chunk = []           # initialise a new chunk
      chunk_dim = 0           # reset dimension counter
    end
    push!(vn_chunk, vns[i])       # put the current variable into the current chunk
    chunk_dim += l            # update dimension counter
  end
  push!(vn_chunks,            # push the last chunk
        (vn_chunk, chunk_dim))

  # Chunk-wise forward AD
  for (vn_chunk, chunk_dim) in vn_chunks
    # 1. Set dual part correspondingly
    @debug "set dual..."
    dim_count = 1
    for i = 1:vn_num
      range = getrange(vi, vns[i])
      l = length(range)
      vals = getval(vi, vns[i])
      if vns[i] in vn_chunk        # for each variable to compute gradient in this round
        for i = 1:l
          vi[range[i]] = ForwardDiff.Dual{Nothing, Float64, CHUNKSIZE}(realpart(vals[i]), SEEDS[dim_count])
          dim_count += 1      # count
        end
      else                    # for other varilables (no gradient in this round)
        for i = 1:l
          vi[range[i]] = ForwardDiff.Dual{Nothing, Float64, CHUNKSIZE}(realpart(vals[i]))
        end
      end
    end
    @debug "set dual done"

    # 2. Run model
    @debug "run model..."
    vi = runmodel(model, vi, spl)

    # 3. Collect gradient
    @debug "collect gradients from logp..."
    append!(grad, collect(dualpart(-getlogp(vi)))[1:chunk_dim])
  end

  if spl != nothing && haskey(spl.info, :grad_cache)
    spl.info[:grad_cache][θ_hash] = grad
  end

  grad
end

verifygrad(grad::Vector{Float64}) = begin
  if any(isnan.(grad)) || any(isinf.(grad))
    dwarn(0, "Numerical error has been found in gradients.")
    dwarn(1, "grad = $(grad)")
    false
  else
    true
  end
end

# Direct call of ForwardDiff.gradient; this is slow

gradient2(_vi::VarInfo, model::Function, spl::Union{Nothing, Sampler}) = begin

  vi = deepcopy(_vi)

  f(x::Vector) = begin
    vi[spl] = x
    -getlogp(runmodel(model, vi, spl))
  end

  g = x -> ForwardDiff.gradient(f, x::Vector, ForwardDiff.GradientConfig{min(length(x),CHUNKSIZE)}(x::Vector))

  g(vi[spl])
end

#  @init @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" begin

gradient_r(theta::Vector{Float64}, vi::VarInfo, model::Function) = gradient_r(theta, vi, model, nothing)
gradient_r(theta::Vector{Float64}, vi::Turing.VarInfo, model::Function, spl::Union{Nothing, Sampler}) = begin
    f_r(ipts) = begin
      vi[spl] = ipts
      -runmodel(model, vi, spl).logp
    end

    grad = Tracker.gradient(f_r, theta)

    # grad = ReverseDiff.gradient(x -> (vi[spl] = x; -runmodel(model, vi, spl).logp), inputs)

    # vi[spl] = realpart(vi[spl])
    # vi.logp = 0

    map(x -> isa(x, Tracker.TrackedReal) ? x.data : x, grad)
end

#  end
