import logging
import time
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [5, 10, 30]  # seconds between retries


class BaseAgent:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def _call(self, *, max_tokens: int = 2048, **kwargs: Any) -> anthropic.types.Message:
        """Call the Claude API with retry on rate-limit and transient errors."""
        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                logger.warning("Retrying Claude API call in %ds (attempt %d)", delay, attempt)
                time.sleep(delay)
            try:
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except anthropic.RateLimitError:
                if attempt == len(_RETRY_DELAYS):
                    raise
                logger.warning("Rate limit hit (attempt %d/%d)", attempt + 1, len(_RETRY_DELAYS) + 1)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500 and attempt < len(_RETRY_DELAYS):
                    logger.warning("Server error %s (attempt %d)", e.status_code, attempt + 1)
                else:
                    raise

        raise RuntimeError("Claude API call failed after all retries")

    def _extract_tool_input(self, response: anthropic.types.Message) -> dict:
        """Extract the input dict from the first tool_use block in a response."""
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        raise ValueError(
            f"No tool_use block in response. Stop reason: {response.stop_reason}. "
            f"Content: {response.content}"
        )
