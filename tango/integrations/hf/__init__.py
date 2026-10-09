"""
.. important::
    To use this integration install this fork with the "hf" extra, from a release wheel
    (``pip install 'ai2_tango[hf] @ https://github.com/anbasile/tango/releases/download/v2.2.0/ai2_tango-2.2.0-py3-none-any.whl'``)
    or from a checkout (``pip install -e '.[hf]'``), or just install ``huggingface_hub`` and
    ``boto3`` after the fact. The fork is not on PyPI: ``pip install ai2-tango[hf]`` fetches
    the original project, which has no such extra.

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

A fuller ``executor`` block, with the settings that matter once runs get long:

.. code:: yaml

    executor:
      type: hf
      image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
      parallelism: 8
      timeout: 4h            # per step; enforced by the executor as well as the platform
      attempts: 2            # submit again a job that dies without the step having failed
      driver_timeout: 2d     # for detached runs: the driver has to outlive the whole graph
      env:
        OMP_NUM_THREADS: 1
      secrets_from_env:
        - WANDB_API_KEY
      extra_project_exclude:
        - "data/**"

Before paying for anything
--------------------------

``tango run --dry-run`` lists the steps that have no result yet, the hardware each would run
on, its price per hour and what it costs at most (if it runs into ``timeout``). Nothing is
registered, uploaded or started::

    $ tango run config.jsonnet -n main --dry-run
    2 of 14 steps would run
      features-pets  [job on t4-small, $0.40/h, at most $1.60; incomplete]
      sweep-pets  [job on cpu-upgrade, $0.03/h, at most $0.12; incomplete; IDENTITY CHANGED since run 'main']

A step's identity is a hash of its arguments, so adding an argument to a step, or changing a
default, gives every such step a new identity, and the next run pays for all of them again.
With ``-n`` of an existing run, the dry run points those out. ``--expect PATTERN`` (repeatable)
turns it into a gate: if a step that matches no pattern would run, nothing is started and the
command fails. Without ``--dry-run`` the run goes ahead once the check passes::

    tango run config.jsonnet -n main --expect 'sweep-*' --executor-option detach=true

``--executor-option KEY=VALUE`` overrides one executor setting for one run; ``detach=true``
hands the graph to a driver job, so the machine that launched it can be switched off.

What a run leaves in the bucket
-------------------------------

Beside step results and run records:

- ``logs/<step unique id>/<job id>.log``: the output of each job, uploaded every five minutes
  and when the job ends. The Hub itself returns about the last thousand lines of a job's log,
  and sometimes none. A driver job's log is under ``logs/_driver/``.
- ``jobs/<run name>/<job id>.json``: one record per job with its step, flavor, price, the
  times it was submitted, started and ended, its final stage and an estimated cost. The Hub
  reports no running time for cancelled jobs and drops old jobs from its list.

When a job dies
---------------

A job can end without the step having finished: cancelled, evicted (for instance when it fills
its disk, 50 GB on ``t4-small``, which leaves no message in the log), killed at its timeout, or
exiting cleanly without a result. The executor then fails the step, records the reason in the
step's info, and removes the lock the job held, so the next run can simply try again. A job is
only trusted to have succeeded when the step's result is in the bucket.

Inside a job
------------

Every job has ``TANGO_HF_JOB=1`` and ``TANGO_HF_FLAVOR`` in its environment, next to the
platform's own ``JOB_ID``. See :class:`~tango.integrations.hf.executor.HfJobsExecutor` for the
other defaults (thread counts, logging, progress bars) and how to override them with ``env``.

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
