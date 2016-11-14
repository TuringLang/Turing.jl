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

