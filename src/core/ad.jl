doc"""
    gradient(vi::VarInfo, model::Function, spl::Union{Void, Sampler})

Function to generate the gradient dictionary, with each prior map to its derivative of the logjoint probibilioty. This function uses chunk-wise forward AD with a chunk of size $(CHUNKSIZE) as default.

Example:

```julia
grad = gradient(vi, model, spl)
end
```
"""
gradient(vi::VarInfo, model::Function) = gradient2(vi, model, nothing)
gradient(_vi::VarInfo, model::Function, spl::Union{Void, Sampler}) = begin

  vi = deepcopy(_vi)

  f(x::Vector) = begin
    vi[spl] = x
    -getlogp(runmodel(model, vi, spl))
  end

  g = x -> ForwardDiff.gradient(f, x::Vector, ForwardDiff.GradientConfig{min(length(x),CHUNKSIZE)}(x::Vector))

  g(vi[spl])
end

gradient2(vi::VarInfo, model::Function, spl::Union{Void, Sampler}) = begin

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
  vn_chunks = []; vn_chunk = []; chunk_dim = 0

  vns_all = getvns(vi, spl)
  for k in vns_all
    l = length(getrange(vi, k))         # dimension for the current variable
    if chunk_dim + l > CHUNKSIZE
      push!(vn_chunks, # store the previous chunk
            (vn_chunk, chunk_dim))
      vn_chunk = []          # initialise a new chunk
      chunk_dim = 0           # reset dimension counter
    end
    push!(vn_chunk, k)       # put the current variable into the current chunk
    chunk_dim += l            # update dimension counter
  end
  push!(vn_chunks,     # push the last chunk
        (vn_chunk, chunk_dim))

  # Chunk-wise forward AD
  for (vn_chunk, chunk_dim) in vn_chunks
    # 1. Set dual part correspondingly
    dprintln(4, "set dual...")
    # dps = zeros(chunk_dim)

    dim_count = 1
    for k in vns_all
      range = getrange(vi, k)
      l = length(range)
      reals = realpart(getval(vi, k))
      if k in vn_chunk         # for each variable to compute gradient in this round
        dprintln(5, "making dual...")
        for i = 1:l
          # dps[dim_count] = 1  # set dual part
          vi[range[i]] = ForwardDiff.Dual{chunk_dim, Float64}(reals[i], prealloc_duals[dim_count])
          # dps[dim_count] = 0  # reset dual part
          dim_count += 1      # count
        end
        dprintln(5, "make dual done")
      else                      # for other varilables (no gradient in this round)
        for i = 1:l
          vi[range[i]] = ForwardDiff.Dual{chunk_dim, Float64}(reals[i])
        end
      end
    end

    # 2. Run model and collect gradient
    vi = runmodel(model, vi, spl)
    dprintln(4, "collect gradients from logp...")
    append!(grad, collect(dualpart(-getlogp(vi))))
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
