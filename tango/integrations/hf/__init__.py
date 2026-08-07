"""
.. important::
    To use this integration you should install ``tango`` with the "hf" extra
    (e.g. ``pip install tango[hf]``) or just install ``huggingface_hub`` and ``boto3``
    after the fact.

Components for Tango integration with the `Hugging Face Hub <https://huggingface.co/docs/hub/>`_:
a :class:`~tango.workspace.Workspace` backed by a
`Storage Bucket <https://huggingface.co/docs/hub/storage-buckets>`_, and an
:class:`~tango.executor.Executor` that runs each step as a
`Job <https://huggingface.co/docs/hub/jobs>`_.

Setup
-----

Log in with ``hf auth login``, then create a bucket for the workspace::

    hf buckets create my-workspace

Steps are locked with a conditional write through the Hugging Face S3 gateway, which uses
credentials separate from your token. Generate them at
https://huggingface.co/settings/tokens ("Generate S3 credentials" on a write token) and export
them::

    export HF_S3_ACCESS_KEY_ID=HFAK...
    export HF_S3_SECRET_ACCESS_KEY=...

Using the workspace on its own
------------------------------

The workspace is useful without the executor: steps run locally while their results are cached
in the bucket, so another machine can reuse them.

.. code-block::

    tango run config.jsonnet -w hf://buckets/my-org/my-workspace

Running steps on Hugging Face hardware
--------------------------------------

Add the executor to your ``tango.yml``:

.. code:: yaml

    workspace:
      type: hf
      bucket: my-org/my-workspace

    executor:
      type: hf
      image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
      parallelism: 4

Each step becomes its own Job. To ask for a GPU, declare it on the step and the executor picks
the cheapest flavor that fits:

.. code:: json

    "steps": {
        "train": {
            "type": "torch::train",
            "step_resources": {"gpu_count": 1}
        }
    }

.. tip::
    Every Job pays a container cold start, so sending many small steps to the cluster is
    wasteful. Set ``"step_resources": {"machine": "local"}`` on the cheap ones to run them
    on your own machine instead.

"""

from tango.common.exceptions import IntegrationMissingError

try:
    import huggingface_hub  # noqa: F401
except ModuleNotFoundError:
    raise IntegrationMissingError("hf", dependencies={"huggingface_hub"})

from .common import Flavor, HfBucketClient, HfStepLock, resolve_flavor
from .endpoint import EndpointBatchStep
from .executor import HfJobsExecutor
from .step_cache import HfBucketStepCache
from .workspace import HfBucketWorkspace

__all__ = [
    "EndpointBatchStep",
    "Flavor",
    "HfBucketClient",
    "HfBucketStepCache",
    "HfBucketWorkspace",
    "HfJobsExecutor",
    "HfStepLock",
    "resolve_flavor",
]
