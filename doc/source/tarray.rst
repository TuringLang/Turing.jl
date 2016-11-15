TArray
=========

.. function:: TArray{T}(dims, ...)

   Implementation of data structures that automatically perform copy-on-write after task copying.

   If current_task is an existing key in ``s``\ , then return ``s[current_task]``\ . Otherwise, return ``s[current_task] = s[last_task]``\ .

   Usage:

   .. code-block:: julia

       TArray(dim)

   Example:

   .. code-block:: julia

       ta = TArray(4)              # init
       for i in 1:4 ta[i] = i end  # assign
       Array(ta)                   # convert to 4-element Array{Int64,1}: [1, 2, 3, 4]

.. function::  tzeros(dims, ...)

   Construct a distributed array of zeros. Trailing arguments are the same as those accepted by ``TArray``\ .

   .. code-block:: julia

       tzeros(dim)

   Example:

   .. code-block:: julia

       tz = tzeros(4)              # construct
       Array(tz)                   # convert to 4-element Array{Int64,1}: [0, 0, 0, 0]

