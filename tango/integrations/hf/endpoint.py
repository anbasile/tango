import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

from tango.common.exceptions import ConfigurationError
from tango.step import Step

logger = logging.getLogger(__name__)


@Step.register("hf::endpoint_batch")
class EndpointBatchStep(Step):
    """
    Run a batch of prompts through a
    `Hugging Face Inference Endpoint <https://huggingface.co/docs/inference-endpoints/>`_,
    creating and tearing down the endpoint around the work.

    .. tip::
        Registered as a :class:`~tango.step.Step` under the name "hf::endpoint_batch".

    .. important::
        A running endpoint costs money for as long as it is up. This step pauses the endpoint
        in a ``finally`` block, so it comes down even if generation fails part-way. It will
        *not* come down if the process is killed outright — check
        https://ui.endpoints.huggingface.co if a run dies badly.

    .. note::
        :attr:`DETERMINISTIC` is ``False``: a decoder with a non-zero temperature returns
        different text each call, so Tango will warn that the cached result isn't reproducible.
        That warning is accurate. Pass ``{"do_sample": false}`` in ``generation_kwargs`` for
        greedy decoding if you need it to be stable.

    :examples:

    .. code:: json

        {
            "type": "hf::endpoint_batch",
            "prompts": ["Label this: ...", "Label this: ..."],
            "endpoint_name": "annotation-endpoint",
            "repository": "Qwen/Qwen3-0.6B",
            "instance_size": "x1",
            "instance_type": "nvidia-a10g",
            "generation_kwargs": {"max_new_tokens": 16, "do_sample": false}
        }
    """

    DETERMINISTIC = False
    CACHEABLE = True

    def _get_or_create(self, kwargs: Dict[str, Any]) -> Any:
        from huggingface_hub import create_inference_endpoint, get_inference_endpoint
        from huggingface_hub.errors import HfHubHTTPError

        name = kwargs["name"]
        namespace = kwargs.get("namespace")
        token = kwargs.get("token")
        try:
            endpoint = get_inference_endpoint(name, namespace=namespace, token=token)
            self.logger.info("Reusing existing endpoint '%s' (%s).", name, endpoint.status)
            return endpoint
        except HfHubHTTPError:
            if not kwargs.get("repository"):
                raise ConfigurationError(
                    f"Inference endpoint '{name}' does not exist and no 'repository' was given "
                    f"to create it with."
                )
        self.logger.info("Creating endpoint '%s' for %s...", name, kwargs["repository"])
        return create_inference_endpoint(
            **{key: value for key, value in kwargs.items() if value is not None}
        )

    async def _generate_all(
        self,
        endpoint: Any,
        prompts: Sequence[str],
        method: str,
        generation_kwargs: Dict[str, Any],
        max_concurrency: int,
    ) -> List[str]:
        client = endpoint.async_client
        semaphore = asyncio.Semaphore(max_concurrency)

        async def one(prompt: str) -> str:
            async with semaphore:
                if method == "chat_completion":
                    response = await client.chat_completion(
                        messages=[{"role": "user", "content": prompt}], **generation_kwargs
                    )
                    return response.choices[0].message.content or ""
                return await client.text_generation(prompt, **generation_kwargs)

        return list(await asyncio.gather(*(one(prompt) for prompt in prompts)))

    def run(  # type: ignore[override]
        self,
        prompts: Sequence[str],
        endpoint_name: str,
        repository: Optional[str] = None,
        method: str = "text_generation",
        framework: str = "pytorch",
        task: Optional[str] = "text-generation",
        accelerator: str = "gpu",
        instance_size: Optional[str] = None,
        instance_type: Optional[str] = None,
        vendor: str = "aws",
        region: str = "us-east-1",
        endpoint_type: str = "protected",
        custom_image: Optional[Dict[str, Any]] = None,
        scale_to_zero_timeout: Optional[int] = None,
        namespace: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 8,
        pause_when_done: bool = True,
        timeout: Optional[float] = 1800.0,
        token: Optional[str] = None,
    ) -> List[str]:
        """
        :param prompts: The prompts to send.
        :param endpoint_name: Name of the endpoint. Reused if it already exists.
        :param repository: Model repo to deploy. Required only when creating a new endpoint.
        :param method: ``"text_generation"`` or ``"chat_completion"``.
        :param max_concurrency: How many requests to keep in flight.
        :param pause_when_done: Pause the endpoint afterwards so it stops costing money.
        :param timeout: Seconds to wait for the endpoint to come up.
        :returns: One generated string per prompt, in the order given.
        """
        if method not in {"text_generation", "chat_completion"}:
            raise ConfigurationError(
                f"'method' must be 'text_generation' or 'chat_completion', got '{method}'."
            )

        endpoint = self._get_or_create(
            {
                "name": endpoint_name,
                "repository": repository,
                "framework": framework,
                "task": task,
                "accelerator": accelerator,
                "instance_size": instance_size,
                "instance_type": instance_type,
                "vendor": vendor,
                "region": region,
                "type": endpoint_type,
                "custom_image": custom_image,
                "scale_to_zero_timeout": scale_to_zero_timeout,
                "namespace": namespace,
                "token": token,
            }
        )

        try:
            if endpoint.status == "paused":
                self.logger.info("Resuming endpoint '%s'...", endpoint_name)
                endpoint = endpoint.resume()
            endpoint = endpoint.wait(timeout=timeout)

            self.logger.info(
                "Sending %d prompts to '%s' with concurrency %d...",
                len(prompts),
                endpoint_name,
                max_concurrency,
            )
            return asyncio.run(
                self._generate_all(
                    endpoint,
                    prompts,
                    method,
                    dict(generation_kwargs or {}),
                    max_concurrency,
                )
            )
        finally:
            if pause_when_done:
                # In a `finally` on purpose: an endpoint left running after a failed step keeps
                # billing by the hour.
                try:
                    endpoint.pause()
                    self.logger.info("Paused endpoint '%s'.", endpoint_name)
                except Exception:
                    self.logger.error(
                        "Failed to pause endpoint '%s'. It may still be running and billing — "
                        "check https://ui.endpoints.huggingface.co",
                        endpoint_name,
                        exc_info=True,
                    )
