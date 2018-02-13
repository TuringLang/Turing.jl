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

  θ_hash = hash(vi[spl])

  if spl != nothing && haskey(spl.info, :grad_cache)
    if haskey(spl.info[:grad_cache], θ_hash)
      return spl.info[:grad_cache][θ_hash]
    end
  end

  # Initialisation
  grad = Vector{Float64}()

  # Split keys(vi) into chunks,
  dprintln(4, "making chunks...")
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
    dprintln(4, "set dual...")
    dim_count = 1
    for i = 1:vn_num
      range = getrange(vi, vns[i])
      l = length(range)
      vals = getval(vi, vns[i])
      if vns[i] in vn_chunk        # for each variable to compute gradient in this round
        for i = 1:l
          vi[range[i]] = ForwardDiff.Dual{Void, Float64, CHUNKSIZE}(realpart(vals[i]), SEEDS[dim_count])
          dim_count += 1      # count
        end
      else                    # for other varilables (no gradient in this round)
        for i = 1:l
          vi[range[i]] = ForwardDiff.Dual{Void, Float64, CHUNKSIZE}(realpart(vals[i]))
        end
      end
    end
    dprintln(4, "set dual done")

    # 2. Run model
    dprintln(4, "run model...")
    vi = runmodel(model, vi, spl)

    # 3. Collect gradient
    dprintln(4, "collect gradients from logp...")
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

gradient2(_vi::VarInfo, model::Function, spl::Union{Void, Sampler}) = begin

  vi = deepcopy(_vi)

  f(x::Vector) = begin
    vi[spl] = x
    -getlogp(runmodel(model, vi, spl))
  end

  g = x -> ForwardDiff.gradient(f, x::Vector, ForwardDiff.GradientConfig{min(length(x),CHUNKSIZE)}(x::Vector))

  g(vi[spl])
end
gradient_t(theta::Vector{Float64}, vi::VarInfo, model::Function) = gradient_t(vi, model, nothing)
gradient_t(theta::Vector{Float64}, vi::Turing.VarInfo, model::Function, spl::Union{Void, Sampler}) = begin
    inputs = (theta)

    if model in RD_CACHE
        f_tape2 = GradientTape(x -> (vi[spl] = x; runmodel(model, vi, spl).logp), inputs)
        compiled_f_tape2 = compile(f_tape2)
        results = (similar(theta))

        RD_CACHE[model] = Dict()
        RD_CACHE[model][:ctape] = compiled_f_tape2
        RD_CACHE[model][:res] = results
    else
        compiled_f_tape2 = RD_CACHE[model][:ctape]
        results = RD_CACHE[model][:res]
    end

    grad = gradient!(results, compiled_f_tape2, inputs)

    vi[spl] = realpart(vi[spl])
    vi.logp = 0
    grad
end