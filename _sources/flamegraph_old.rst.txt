Flame graph of original implementation
======================================

Most computation time is spent on line 85--88:

.. code-block:: python

   ...
   I_ = np.array(I)
   J_ = np.array(J)
   K_ = np.maximum(I_, J_)
   G[u, v] = eta[tuple(K_)] - eta[I] * eta[J]

Right click + 'Open image in new tab' to see timing details.

.. image:: images/profile_old.svg