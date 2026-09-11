"""
Gemini LLM Client — thin wrapper around the Google GenAI SDK.

Provides text generation for:
  - LLM-as-a-Judge rubric evaluation
  - Grounded reply drafting
  - Graceful degradation when no API key is set

Requires GEMINI_API_KEY environment variable.
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Lazy-initialised client — avoids import-time failure when key is absent
_client = None
_initialised = False

GEMINI_MODEL = "gemini-2.0-flash"


def _ensure_client():
    """Lazily initialise the google-genai client once."""
    global _client, _initialised
    if _initialised:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "GEMINI_API_KEY not set — LLM features will fall back to heuristic mode. "
            "Get a free key at https://aistudio.google.com/"
        )
        _initialised = True
        _client = None
        return None

    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        _initialised = True
        logger.info("Gemini client initialised (model=%s)", GEMINI_MODEL)
    except ImportError:
        logger.warning("google-genai package not installed — pip install google-genai")
        _initialised = True
        _client = None
    except Exception as exc:
        logger.warning("Failed to initialise Gemini client: %s", exc)
        _initialised = True
        _client = None

    return _client


def is_available() -> bool:
    """Return True if the Gemini API is configured and reachable."""
    return _ensure_client() is not None


def generate(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    temperature: float = 0.3,
    max_output_tokens: int = 1024,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Generate text from the Gemini model.

    Returns the generated text string, or None if the API is unavailable
    or all retries are exhausted.
    """
    client = _ensure_client()
    if client is None:
        return None

    from google import genai
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if system_instruction:
        config.system_instruction = system_instruction

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            if response and response.text:
                return response.text.strip()
            return None
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(
                "Gemini API call failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)

    logger.error("All %d Gemini API retries exhausted", max_retries)
    return None


def generate_json(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
) -> Optional[Dict[str, Any]]:
    """
    Generate a JSON object from the Gemini model.

    Prompts the model to return valid JSON. Parses and returns the dict,
    or None on failure.
    """
    raw = generate(
        prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if raw is None:
        return None

    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini response as JSON: %s", text[:200])
        return None
