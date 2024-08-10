# Copyright (c) 2024 Oracle Corporation.
# Licensed under the MIT License

"""GenerativeAI Configuration class definition."""

import json
from collections.abc import Hashable
from typing import Any, cast

from graphrag.llm.types import LLMConfig


def _non_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return None if stripped == "" else value


class OCIGenAIConfiguration(Hashable, LLMConfig):
    """GenerativeAI Configuration class definition."""

    # Core Configuration
    _config_profile: str
    _compartment_id: str
    _endpoint: str
    _model: str | None
    _model_id: str

    # Operation Configuration
    _truncate: str | None
    _n: int | None
    _temperature: float | None
    _frequency_penalty: float | None
    _presence_penalty: float | None
    _top_p: float | None
    _top_k: int | None
    _max_tokens: int | None
    _stop: list[str] | None

    # Retry Logic
    _max_retries: int | None
    _request_timeout: tuple[int, int] | None
    # Feature Flags
    _model_supports_json: bool | None
    _concurrent_requests: int | None

    # The raw configuration object
    _raw_config: dict

    def __init__(
        self,
        config: dict,
    ):
        """Init method definition."""

        def lookup_required(key: str) -> str:
            return cast(str, config.get(key))

        def lookup_str(key: str) -> str | None:
            return cast(str | None, config.get(key))

        def lookup_int(key: str) -> int | None:
            result = config.get(key)
            if result is None:
                return None
            return int(cast(int, result))

        def lookup_float(key: str) -> float | None:
            result = config.get(key)
            if result is None:
                return None
            return float(cast(float, result))

        def lookup_list(key: str) -> list | None:
            return cast(list | None, config.get(key))

        def lookup_bool(key: str) -> bool | None:
            value = config.get(key)
            if isinstance(value, str):
                return value.upper() == "TRUE"
            if isinstance(value, int):
                return value > 0
            return cast(bool | None, config.get(key))

        self._config_profile = lookup_required("config_profile")
        self._compartment_id = lookup_required("compartment_id")
        self._endpoint = lookup_required("endpoint")
        self._model = lookup_required("model")
        self._model_id = lookup_required("model_id")
        self._truncate = lookup_str("truncate")
        self._n = lookup_int("n")
        self._temperature = lookup_float("temperature")
        self._frequency_penalty = lookup_float("frequency_penalty")
        self._presence_penalty = lookup_float("presence_penalty")
        self._top_p = lookup_float("top_p")
        self._top_k = lookup_int("top_k")
        self._max_tokens = lookup_int("max_tokens")
        self._stop = lookup_list("stop")
        self._max_retries = lookup_int("max_retries")
        self._request_timeout = lookup_float("request_timeout")
        self._model_supports_json = lookup_bool("model_supports_json")
        self._concurrent_requests = lookup_int("concurrent_requests")
        self._raw_config = config

    @property
    def config_profile(self) -> str:
        """Config profile property definition."""
        return self._config_profile

    @property
    def compartment_id(self) -> str:
        """Compartment ID property definition."""
        return self._compartment_id

    @property
    def endpoint(self) -> str:
        """Endpoint property definition."""
        return self._endpoint

    @property
    def model(self) -> str:
        """Model ID property definition."""
        return self._model

    @property
    def model_id(self) -> str:
        """Model ID property definition."""
        return self._model_id

    @property
    def truncate(self) -> str | None:
        """Truncate property definition."""
        return _non_blank(self._truncate)

    @property
    def n(self) -> int | None:
        """N property definition."""
        return self._n

    @property
    def temperature(self) -> float | None:
        """Temperature property definition."""
        return self._temperature

    @property
    def frequency_penalty(self) -> float | None:
        """Top p property definition."""
        return self._frequency_penalty

    @property
    def presence_penalty(self) -> float | None:
        """Top p property definition."""
        return self._presence_penalty

    @property
    def top_p(self) -> float | None:
        """Top p property definition."""
        return self._top_p

    @property
    def top_k(self) -> int | None:
        """Top p property definition."""
        return self._top_k

    @property
    def max_tokens(self) -> int | None:
        """Max tokens property definition."""
        return self._max_tokens

    @property
    def stop(self) -> list[str] | None:
        """Stop property definition."""
        return self._stop

    @property
    def max_retries(self) -> int | None:
        """Max retries property definition."""
        return self._max_retries

    @property
    def request_timeout(self) -> tuple[int, int] | None:
        """Request timeout property definition."""
        return self._request_timeout

    @property
    def model_supports_json(self) -> bool | None:
        """Model supports json property definition."""
        return self._model_supports_json

    @property
    def concurrent_requests(self) -> int | None:
        """Concurrent requests property definition."""
        return self._concurrent_requests

    @property
    def raw_config(self) -> dict:
        """Raw config method definition."""
        return self._raw_config

    def lookup(self, name: str, default_value: Any = None) -> Any:
        """Lookup method definition."""
        return self._raw_config.get(name, default_value)

    def __str__(self) -> str:
        """Str method definition."""
        return json.dumps(self.raw_config, indent=4)

    def __repr__(self) -> str:
        """Repr method definition."""
        return f"GenerativeAIConfiguration({self._raw_config})"

    def __eq__(self, other: object) -> bool:
        """Eq method definition."""
        if not isinstance(other, OCIGenAIConfiguration):
            return False
        return self._raw_config == other._raw_config

    def __hash__(self) -> int:
        """Hash method definition."""
        return hash(tuple(sorted(self._raw_config.items())))
