``inference`` – Inference algorithms
------------------------------------

Finally, given a generator instance, we want to do parameter inference.

Inference algorithms are implemented in the inference submodule. At the moment,
the following algorithms are implemented:
- ``Basic``
- ``SNPE``
- ``CDE-LFI``

Basic
`````
.. autoclass:: delfi.inference.Basic
  :show-inheritance:
  :inherited-members:
  :members:

SNPE
````
.. autoclass:: delfi.inference.SNPE
  :show-inheritance:
  :inherited-members:
  :members:

CDELFI
``````
.. autoclass:: delfi.inference.CDELFI
  :show-inheritance:
  :inherited-members:
  :members:
