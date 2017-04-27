doc"""
    gradient(spl :: GradientSampler)

Function to generate the gradient dictionary, with each prior map to its derivative of the logjoint. This function uses chunk-wise forward AD with a chunk of size 10, which is limited by the ForwardDiff package.

Example:

```julia
function Base.run(spl :: Sampler{HMC})
  ...
  val∇E = gradient(spl.priors, spl.model)
  ...
end
```
"""
function gradient(_vi::VarInfo, model::Function, spl=nothing)

  vi = deepcopy(_vi)
  # Initialisation
  val∇E = Dict{Tuple, Vector{Float64}}()

  # Split keys(values) into CHUNKSIZE, CHUNKSIZE, CHUNKSIZE, m-size chunks,
  dprintln(4, "making chunks...")
  prior_key_chunks = []
  key_chunk = []
  prior_dim = 0

  gkeys = keys(vi)
  if spl != nothing && !isempty(spl.alg.space)
    gkeys = filter(k -> getgid(vi, k) == spl.alg.group_id || (getgid(vi, k) == 0 && getsym(vi, k) in spl.alg.space), keys(vi))
  end

  for k in gkeys
    l = length(vi[k])
    if prior_dim + l > CHUNKSIZE
      # Store the old chunk
      push!(prior_key_chunks, (key_chunk, prior_dim))
      # Initialise new chunk
      key_chunk = []
      prior_dim = 0
      # Update
      push!(key_chunk, k)
      prior_dim += l
    else
      # Update
      push!(key_chunk, k)
      prior_dim += l
    end
  end
  if length(key_chunk) != 0
    push!(prior_key_chunks, (key_chunk, prior_dim))  # push the last chunk
  end

  # chunk-wise forward AD
  for (key_chunk, prior_dim) in prior_key_chunks
    # Set dual part correspondingly
    dprintln(4, "set dual...")
    dps = zeros(prior_dim)
    prior_count = 1
    for k in gkeys
      l = length(vi[k])
      reals = realpart(vi[k])
      val_vect = vi[k]        # get a reference for the value vector
      if k in key_chunk       # to graidnet variables
        dprintln(5, "making dual...")
        for i = 1:l
          dps[prior_count] = 1  # set dual part
          val_vect[i] = Dual(reals[i], dps...)
          dps[prior_count] = 0  # reset dual part
          prior_count += 1      # count
        end
        dprintln(5, "make dual done")
      else                    # other varilables (not for gradient info)
        for i = 1:l           # NOTE: we cannot use direct assignment here as we dont' want the reference of val_vect is changed (Mv and Mat support)
          val_vect[i] = Dual{prior_dim, Float64}(reals[i])
        end
      end
    end
    # Run the model
    dprintln(4, "run model...")
    vi.logjoint = Dual{prior_dim, Float64}(0)
    vi = runmodel(model, vi, spl)
    # Collect gradient
    dprintln(4, "collect dual...")
    prior_count = 1
    for k in key_chunk
      dprintln(5, "for each prior...")
      l = length(vi[k])
      duals = dualpart(-vi.logjoint)
      # To store the gradient vector
      g = zeros(l)
      for i = 1:l # NOTE: we cannot use direct assignment here as we dont' want the reference of val_vect is changed (Mv and Mat support)
        dprintln(5, "taking from logjoint...")
        g[i] = duals[prior_count] # collect
        prior_count += 1          # count
      end
      val∇E[k] = g
    end
    # Reset logjoint
    vi.logjoint = Dual{prior_dim, Float64}(0)
  end
  # Return
  return val∇E
end

verifygrad(grad::Dict) = begin
  valid = true
  for k in keys(grad)
    if any(isnan(grad[k])) || any(isinf(grad[k]))
      dwarn(0, "NaN/Inf gradients")
      dwarn(1, "grad = $(grad)")
      valid = false
    end
  end
  valid
end
