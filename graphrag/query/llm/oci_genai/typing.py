# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OpenAI wrapper options."""

from enum import Enum


class OCIGenAIApiType(str, Enum):
    """The OCIGenAI Flavor."""

    OCIGenAI = "oci_genai"
