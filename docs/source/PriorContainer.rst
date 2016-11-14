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

