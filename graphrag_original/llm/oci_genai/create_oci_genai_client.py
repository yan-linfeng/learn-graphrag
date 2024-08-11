# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Create OpenAI client instance."""

import logging
from functools import cache

import oci
from oci.retry import NoneRetryStrategy

from .oci_genai_configuration import OCIGenAIConfiguration
from .types import OCIGenAIClientTypes

log = logging.getLogger(__name__)

ENDPOINT_REQUIRED = "endpoint is required for OCI GenerativeAI client"


@cache
def create_oci_genai_client(
    configuration: OCIGenAIConfiguration
) -> OCIGenAIClientTypes:
    """Create a new OpenAI client instance."""
    """Create a new OCI GenerativeAI client instance."""
    if configuration.endpoint is None:
        raise ValueError(ENDPOINT_REQUIRED)

    log.info(
        "Creating OCI GenerativeAI client endpoint=%s, compartment_id=%s",
        configuration.endpoint,
        configuration.compartment_id,
    )

    config = oci.config.from_file('~/.oci/config', configuration.config_profile)

    client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=configuration.endpoint,
        retry_strategy=NoneRetryStrategy(),
        timeout=configuration.request_timeout or (10, 240)
    )

    return client
