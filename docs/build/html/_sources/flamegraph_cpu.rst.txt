Flame graph of CPU implementation
=================================

Most computation time is spent on ``cumsum`` operation inside ``get_eta`` and ``get_h``.
Input tensor size was increased by 2 dimension for easier comparison
with the original implementation

Right click + 'Open image in new tab' to see timing details.

.. image:: images/profile_cpu_XL.svg