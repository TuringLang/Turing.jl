doc"""
    get_gradient_dict(spl :: GradientSampler)

Function to generate the gradient dictionary, with each prior map to its derivative of the logjoint. This function uses chunk-wise forward AD with a chunk of size 10, which is limited by the ForwardDiff package.

Example:

```julia
function Base.run(spl :: Sampler{HMC})
  ...
  val∇E = get_gradient_dict(spl.priors, spl.model)
  ...
end
```
"""
function get_gradient_dict(values::GradientInfo, model::Function, data, spl)
  # Initialisation
  val∇E = Dict{Any, Any}()
  # Split keys(values) into CHUNKSIZE, CHUNKSIZE, CHUNKSIZE, m-size chunks,
  dprintln(4, "making chunks...")
  prior_key_chunks = []
  key_chunk = []
  prior_dim = 0
  for k in keys(values)
    l = length(values[k])
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
    prior_count = 1
    for k in keys(values)
      l = length(values[k])
      reals = realpart(values[k])
      val_vect = values[k]   # get the value vector

      if k in key_chunk   # to graidnet variables
        for i = 1:l
          dprintln(5, "making dual...")
          val_vect[i] = make_dual(prior_dim, reals[i], prior_count)
          dprintln(5, "make dual done")
          # Count
          prior_count += 1
        end
      else                # other varilables
        for i = 1:l
          val_vect[i] = Dual{prior_dim, Float64}(reals[i])
        end
      end
    end
    # Run the model
    dprintln(4, "run model...")
    values = model(data, values, spl)
    # Collect gradient
    dprintln(4, "collect dual...")
    prior_count = 1
    for k in key_chunk
      dprintln(5, "for each prior...")
      l = length(values[k])
      reals = realpart(values[k])
      # To store the gradient vector
      g = zeros(l)
      for i = 1:l
        # Collect
        dprintln(5, "taking from logjoint...")
        g[i] = dualpart(-values.logjoint)[prior_count]
        # Count
        prior_count += 1
      end
      val∇E[k] = g
    end
    # Reset logjoint
    values.logjoint = Dual(0)
  end
  # Return
  return val∇E
end
