"""
FLARE — cached-fallback wrapper for live LLM calls
====================================================
A live interview demo shouldn't visibly break because of venue wifi or a
rate limit. This wrapper tries the real API call first; on any failure it
falls back to the last successful response cached under the same key. On
success it refreshes the cache, so the first real run of a demo seeds its
own fallback for next time.
"""

import json
import re
from pathlib import Path
from typing import Callable

CACHE_DIR = Path(__file__).parent / "demo_cache"
CACHE_DIR.mkdir(exist_ok=True)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def parse_json_response(text: str) -> dict:
    """Parse a model's JSON reply, tolerating ```json ... ``` fence wrapping.

    Models sometimes wrap structured output in markdown fences even when the
    prompt asks for raw JSON; stripping that is more robust than tightening
    the prompt further and hoping.
    """
    text = text.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def call_with_cache(cache_key: str, live_fn: Callable[[], dict]) -> dict:
    """Try a live call; fall back to the cached response for cache_key on failure.

    Returns the response dict with a "_source" field ("live" or "cached")
    added, so callers/notebooks can display which one was used.
    """
    cache_path = CACHE_DIR / f"{cache_key}.json"

    try:
        result = live_fn()
        cache_path.write_text(json.dumps(result, indent=2))
        return {**result, "_source": "live"}
    except Exception as e:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            print(f"    [fallback] live call failed ({e}) — using cached response for '{cache_key}'")
            return {**cached, "_source": "cached"}
        raise RuntimeError(
            f"Live call failed for '{cache_key}' and no cached fallback exists yet: {e}"
        ) from e
