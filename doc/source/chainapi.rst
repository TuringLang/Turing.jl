Chain
=========

.. function:: Chain(weight::Float64, value::Array{Sample})

   A wrapper of output trajactory of samplers.

   Example:

   .. code-block:: julia

       # Define a model
       @model xxx begin
         ...
         @predict mu sigma
       end

       # Run the inference engine
       chain = sample(xxx, SMC(1000))

       chain[:logevidence]   # show the log model evidence
       chain[:mu]            # show the weighted trajactory for :mu
       chain[:sigma]         # show the weighted trajactory for :sigma
       mean(chain[:mu])      # find the mean of :mu
       mean(chain[:sigma])   # find the mean of :sigma

.. function:: Sample(weight::Float64, value::Dict{Symbol,Any})

   A wrapper of output samples.

   Example:

   .. code-block:: julia

       # Define a model
       @model xxx begin
         ...
         @predict mu sigma
       end

       # Run the inference engine
       chain = sample(xxx, SMC(1000))

       sample = chain[:mu][1]  # get the first sample
       sample.weight           # show the weight of this sample
       sample.value            # show the value of this sample (a dictionary)

