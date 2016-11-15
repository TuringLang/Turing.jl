Replay
=========

.. function:: Prior(sym)

   A wrapper of symbol type representing priors.

   Usage:

   .. code-block:: julia

       p = Prior(:somesym)
       strp = string(p)

.. function:: PriorArray(array, count, currSetIdx, currGetIdx)

   Type for a rotated array (used for prior replay purpose).

   This type of array support set and get without specifying indices. Instead, an inner index pointer is used to iterate the array. The pointers for set and get are separate.

   Usage:

   .. code-block:: julia

       pa = PriorArray() # []
       add(pa, 1)        # [1]
       add(pa, 2)        # [1, 2]
       get(pa)           # 1
       get(pa)           # 2
       set(pa, 3)        # [3, 2]
       get(pa)           # 3
       get(pa)           # 2

.. function:: PriorContainer()

   A container to store priors based on dictionary.

   This type is basically a dictionary supporting adding new priors by creating a PriorArray and indexing using pc[] syntax

   Usage:

   .. code-block:: julia

       pc = PriorContainer()
       p1 = Prior(:a)
       p2 = Prior(:b)

       addPrior(pc, p1, 1)
       addPrior(pc, p1, 2)
       addPrior(pc, p1, 3)
       addPrior(pc, p2, 4)

       pc[p1]    # 1
       pc[p1]    # 2
       pc[p1]    # 3
       pc[p1]    # 1
       pc[p1]    # 2
       pc[p1]    # 3

       pc[p2]    # 4

       pc[p1] = 5
       pc[p1] = 6
       pc[p1] = 7

       pc[p1]    # 5
       pc[p1]    # 6
       pc[p1]    # 7

       keys(pc)  # create a key interator in the container, i.e. all the priors

.. function:: addPrior(pc::PriorContainer, idx::Prior, val)

   Add a *new* value of a given prior to the container. *new* here means force appending to the end of the corresponding array of the prior.

