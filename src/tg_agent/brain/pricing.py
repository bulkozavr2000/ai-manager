"""What a model call costs, from the token usage the API reports."""

from __future__ import annotations

from tg_agent.brain.schemas import Usage

# USD per million tokens: input, output, cache read, cache write (5-minute TTL).
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.50),
    # Answers when the API's server-side fallback replaces Sonnet 5.5; same prices.
    "claude-sonnet-5": (2.0, 10.0, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
}


# Anthropic's web search tool: $10 per 1000 searches, on top of the tokens.
WEB_SEARCH_USD = 0.01


def cost_usd(model: str, usage: Usage) -> float:
    """Dollar cost of one call. An unknown model raises KeyError: never guess a price."""
    p_in, p_out, p_read, p_write = PRICES[model]
    total = (
        usage.input_tokens * p_in
        + usage.output_tokens * p_out
        + usage.cache_read * p_read
        + usage.cache_write * p_write
    )
    return total / 1_000_000


def price(model: str, usage: Usage) -> float:
    """What a call cost. A model missing from the price list is charged by the longest listed
    name it starts with, else at the dearest rate.

    The API may name the model more precisely than the list does (a dated id), or a fallback
    model may have answered: the call has happened, so its cost is counted on the safe side.
    """
    if model in PRICES:
        return cost_usd(model, usage)
    known = [name for name in PRICES if model.startswith(name)]
    if known:
        return cost_usd(max(known, key=len), usage)
    return max(cost_usd(name, usage) for name in PRICES)
