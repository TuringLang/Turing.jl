Compiler
=========

.. function:: @model(name, fbody)

   Wrapper for models.

   Usage:

   .. code-block:: julia

       @model model() = begin
         # body
       end

   Example:

   .. code-block:: julia

       @model gauss() = begin
         s ~ InverseGamma(2,3)
         m ~ Normal(0,sqrt(s))
         1.5 ~ Normal(m, sqrt(s))
         2.0 ~ Normal(m, sqrt(s))
         return(s, m)
       end

.. function:: var_name ~ Distribution()

   ``~`` notation is to specifiy *a variable follows a distributions*. 

   If ``var_name`` is an un-defined variable or a container (e.g. Vector or Matrix), this variable will be treated as model parameter; otherwise if ``var_name`` is defined, this variable will be treated as data.

