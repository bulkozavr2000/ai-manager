from __future__ import annotations

import pytest

from tg_agent.brain.pricing import cost_usd, price
from tg_agent.brain.schemas import Usage

MILLION = 1_000_000


def test_cost_usd_sonnet() -> None:
    assert cost_usd("claude-sonnet-5-5", Usage(MILLION, MILLION)) == pytest.approx(12.0)


def test_cost_usd_cache_read() -> None:
    usage = Usage(0, 0, cache_read=MILLION)
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(0.20)


def test_cost_usd_haiku_and_cache_write() -> None:
    usage = Usage(MILLION, MILLION, cache_write=MILLION)
    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx(0.10 + 0.50 + 0.125)


def test_typical_decision_costs_about_a_cent() -> None:
    assert cost_usd("claude-sonnet-5-5", Usage(4500, 500)) == pytest.approx(0.014)


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        cost_usd("gpt-4", Usage(1, 1))


def test_price_of_a_listed_model() -> None:
    assert price("claude-haiku-5-5", Usage(MILLION, MILLION)) == pytest.approx(0.60)


def test_price_of_a_dated_id_uses_the_listed_name_it_starts_with() -> None:
    usage = Usage(MILLION, MILLION)
    assert price("claude-haiku-5-5-20261001", usage) == pytest.approx(0.60)


def test_price_of_an_unknown_model_is_the_dearest_rate() -> None:
    assert price("gpt-4", Usage(MILLION, MILLION)) == pytest.approx(24.0)


def test_the_server_side_fallback_model_is_priced_as_listed() -> None:
    usage = Usage(MILLION, MILLION, cache_read=MILLION, cache_write=MILLION)
    assert price("claude-sonnet-5", usage) == pytest.approx(2.0 + 10.0 + 0.20 + 2.50)
