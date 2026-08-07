🤗 Hub
======

.. automodule:: tango.integrations.hf

Reference
---------

Workspace
~~~~~~~~~

.. Both classes expose the same ``Constants``, so document it on neither.
.. autoclass:: tango.integrations.hf.HfBucketWorkspace
   :members:
   :exclude-members: Constants

.. autoclass:: tango.integrations.hf.HfBucketStepCache
   :members:
   :exclude-members: Constants

Executor
~~~~~~~~

.. autoclass:: tango.integrations.hf.HfJobsExecutor
   :members:

Steps
~~~~~

.. autoclass:: tango.integrations.hf.EndpointBatchStep
   :members:

Utilities
~~~~~~~~~

.. autoclass:: tango.integrations.hf.HfBucketClient
   :members:

.. autoclass:: tango.integrations.hf.HfStepLock
   :members:

.. autoclass:: tango.integrations.hf.Flavor

.. autofunction:: tango.integrations.hf.resolve_flavor

.. autofunction:: tango.integrations.hf.common.get_s3_client

.. autofunction:: tango.integrations.hf.common.parse_memory

.. autofunction:: tango.integrations.hf.common.split_bucket_path
