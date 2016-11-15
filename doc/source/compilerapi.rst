Macros for Compiler
=========

.. function:: assume(ex)

   Operation for defining the prior.

   Usage:

   .. code-block:: julia

       @assume x ~ Dist

   Here ``x`` is a **symbol** to be used and ``Dist`` is a valid distribution from the Distributions.jl package. Optional parameters can also be passed (see examples below).

   Example:

   .. code-block:: julia

       @assume x ~ Normal(0, 1)
       @assume x ~ Binomial(0, 1)
       @assume x ~ Normal(0, 1; :static=true)
       @assume x ~ Binomial(0, 1; :param=true)

.. function:: observe(ex)

   Operation for defining the likelihood.

   Usage:

   .. code-block:: julia

       @observe x ~ Dist

   Here ``x`` is a **concrete value** to be used and ``Dist`` is a valid distribution from the Distributions.jl package. Optional parameters can also be passed (see examples below).

   Example:

   .. code-block:: julia

       @observe x ~ Normal(0, 1)
       @observe x ~ Binomial(0, 1)
       @observe x ~ Normal(0, 1; :static=true)
       @observe x ~ Binomial(0, 1; :param=true)

.. function:: predict(ex...)

   Operation for defining the the variable(s) to return.

   Usage:

   .. code-block:: julia

       @predict x y z

   Here ``x``\ , ``y``\ , ``z`` are symbols.

.. function:: model(name, fbody)

   Wrapper for models.

   Usage:

   .. code-block:: julia

       @model f body

   Example:

   .. code-block:: julia

       @model gauss begin
         @assume s ~ InverseGamma(2,3)
         @assume m ~ Normal(0,sqrt(s))
         @observe 1.5 ~ Normal(m, sqrt(s))
         @observe 2.0 ~ Normal(m, sqrt(s))
         @predict s m
       end

