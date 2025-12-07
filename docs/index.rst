sparselist
==========

A memory-efficient list subclass that stores only explicitly-set values.

Overview
--------

``sparselist`` is perfect for scenarios where you need a large list but only a
fraction of positions contain meaningful data. Instead of allocating memory for
every position, it uses an internal dictionary to store only the explicitly-set values.

**Materialization** means creating actual Python objects in memory for every list
position. Sparselist avoids this by only storing explicitly-set values: a
size-1,000,000 list with 10 values uses O(10) memory, not O(1,000,000).

Installation
------------

.. code-block:: bash

   pip install sparselist

Quick Example
-------------

.. code-block:: python

   from sparselist import sparselist

   # Create a sparse list of size 1,000,000 with only 3 values
   sl = sparselist({0: 'a', 100: 'b', 1000: 'c'}, size=1000000, default='')

   # Memory efficient: uses O(3) space, not O(1,000,000)
   print(len(sl))  # 1000000
   print(sl[50])   # '' (default)
   print(sl[100])  # 'b'

   # All standard list operations work
   sl.append('d')
   sl.sort()  # O(E log E) where E = explicit count, not O(N log N)!

API Reference
=============

.. autoclass:: sparselist.sparselist
   :members:
   :undoc-members:
   :special-members: __init__, __len__, __reduce_ex__, __setstate__, __getitem__, __setitem__, __delitem__, __iter__, __contains__, __eq__, __ne__, __lt__, __le__, __gt__, __ge__, __add__, __radd__, __iadd__, __mul__, __rmul__, __imul__, __copy__, __repr__, __hash__
   :member-order: bysource
