doc"""
    gradient(vi::VarInfo, model::Function, spl::Union{Void, Sampler})

Function to generate the gradient dictionary, with each prior map to its derivative of the logjoint probibilioty. This function uses chunk-wise forward AD with a chunk of size $(CHUNKSIZE) as default.

Example:

```julia
grad = gradient(vi, model, spl)
end
```
"""
gradient(vi::VarInfo, model::Function) = gradient(vi, model, nothing)
gradient(vi::VarInfo, model::Function, spl::Union{Void, Sampler}) = begin

  θ = realpart(vi[spl])
  if spl != nothing && haskey(spl.info, :grad_cache)
    if haskey(spl.info[:grad_cache], θ)
      return spl.info[:grad_cache][θ]
    end
  end

  # Initialisation
  grad = Vector{Float64}()

  # Split keys(vi) into chunks,
  dprintln(4, "making chunks...")
  vn_chunk = []; vn_chunks = []; chunk_dim = 0;

  vns_all = getvns(vi, spl)
  for vn in vns_all
    l = length(getrange(vi, vn))           # dimension for the current variable
    if chunk_dim + l > CHUNKSIZE
      push!(vn_chunks,        # store the previous chunk
            (vn_chunk, chunk_dim))
      vn_chunk = []           # initialise a new chunk
      chunk_dim = 0           # reset dimension counter
    end
    push!(vn_chunk, vn)       # put the current variable into the current chunk
    chunk_dim += l            # update dimension counter
  end
  push!(vn_chunks,            # push the last chunk
        (vn_chunk, chunk_dim))

  # Chunk-wise forward AD
  for (vn_chunk, chunk_dim) in vn_chunks
    # 1. Set dual part correspondingly
    dprintln(4, "set dual...")
    dim_count = 1
    for vn in vns_all
      range = getrange(vi, vn)
      l = length(range)
      vals = getval(vi, vn)
      if vn in vn_chunk        # for each variable to compute gradient in this round

        for i = 1:l
          vi[range[i]] = ForwardDiff.Dual{CHUNKSIZE, Float64}(vals[i], SEEDS[dim_count])
          dim_count += 1      # count
        end

      else                    # for other varilables (no gradient in this round)
        for i = 1:l
          vi[range[i]] = ForwardDiff.Dual{CHUNKSIZE, Float64}(vals[i])
        end
      end
    end
    dprintln(4, "set dual done")

    # 2. Run model and collect gradient
    dprintln(4, "run model...")
    vi = runmodel(model, vi, spl)
    dprintln(4, "collect gradients from logp...")
    append!(grad, collect(dualpart(-getlogp(vi)))[1:chunk_dim])
  end

  if spl != nothing && haskey(spl.info, :grad_cache)
    spl.info[:grad_cache][θ] = grad
  end

  grad
end

verifygrad(grad::Vector{Float64}) = begin
  if any(isnan(grad)) || any(isinf(grad))
    dwarn(0, "Numerical error has been found in gradients.")
    dwarn(1, "grad = $(grad)")
    false
  else
    true
  end
end

# Direct call of ForwardDiff.gradient; this is slow

gradient2(_vi::VarInfo, model::Function, spl::Union{Void, Sampler}) = begin

  vi = deepcopy(_vi)

  f(x::Vector) = begin
    vi[spl] = x
    -getlogp(runmodel(model, vi, spl))
  end

  g = x -> ForwardDiff.gradient(f, x::Vector, ForwardDiff.GradientConfig{min(length(x),CHUNKSIZE)}(x::Vector))

  g(vi[spl])
end
