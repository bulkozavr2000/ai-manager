"""The Telegram double itself: what it does when told to, so the tests built on it can trust it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tg_agent.clock import FakeClock
from tg_agent.gateway.base import (
    ChatMessage,
    PeerInfo,
    TgActionError,
    TgSendUncertain,
    name_matches,
)
from tg_agent.gateway.fake import FakeGateway
from tg_agent.policy.tg_safety import TgErrorKind

PEER = 555


async def test_replay_records_the_request_and_answers_with_what_it_was_told() -> None:
    gateway = FakeGateway()
    await gateway.set_allowed({PEER})
    gateway.replay_count = 3

    assert await gateway.replay(PEER, 40, {41, 42}) == 3

    assert gateway.replays == [(PEER, 40, {41, 42})]
    assert gateway.calls[-1] == "replay"


async def test_replay_of_a_chat_that_is_not_allowed_reads_nothing() -> None:
    gateway = FakeGateway()
    gateway.replay_count = 3  # there is something to hand over, but not for this chat

    assert await gateway.replay(PEER, 40, set()) == 0

    assert gateway.replays == []


async def test_replay_fails_when_told_to_once() -> None:
    gateway = FakeGateway()
    await gateway.set_allowed({PEER})
    gateway.fail_next = TgActionError(TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT_30", 30)

    with pytest.raises(TgActionError):
        await gateway.replay(PEER, 40, set())
    assert gateway.replays == []
    assert (
        await gateway.replay(PEER, 40, set()) == 0
    )  # the next call is not told to fail


async def test_a_send_can_be_staged_to_end_in_doubt() -> None:
    gateway = FakeGateway()
    gateway.fail_next = TgSendUncertain("ConnectionError")

    with pytest.raises(TgSendUncertain):
        await gateway.send_text(PEER, "Здравствуйте")
    assert gateway.sent == []  # in the double the message did not go out

    assert await gateway.send_text(PEER, "Здравствуйте") == 1  # told to fail once only


# -- the chat log and recent_messages (the agent core reads a chat) --


async def test_a_send_is_written_to_the_chat_log() -> None:
    gateway = FakeGateway()

    first = await gateway.send_text(PEER, "Здравствуйте")
    second = await gateway.send_file(PEER, "photo.jpg")

    log = gateway.chats[PEER]
    assert [(m.tg_msg_id, m.out, m.text) for m in log] == [
        (first, True, "Здравствуйте"),
        (second, True, ""),  # a file has no text
    ]
    assert all(m.date.utcoffset() is not None for m in log)


async def test_add_incoming_scripts_a_counterpart_s_message() -> None:
    gateway = FakeGateway()
    ours = await gateway.send_text(PEER, "Здравствуйте")

    message = gateway.add_incoming(PEER, "Цена 120 000")

    assert isinstance(message, ChatMessage)
    assert (message.out, message.text) == (False, "Цена 120 000")
    assert message.tg_msg_id > ours  # a fresh id
    assert gateway.chats[PEER][-1] == message
    assert gateway.sent == [(PEER, "Здравствуйте")]  # nothing was sent


async def test_with_a_clock_the_chat_log_is_dated_by_it() -> None:
    clock = FakeClock(datetime(2026, 11, 10, 10, 0, tzinfo=UTC))
    gateway = FakeGateway(clock)

    await gateway.send_text(PEER, "Здравствуйте")
    clock.advance(timedelta(minutes=5))
    await gateway.send_file(PEER, "photo.jpg")
    clock.advance(timedelta(minutes=5))
    gateway.add_incoming(PEER, "Цена 120 000")

    start = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
    assert [m.date for m in gateway.chats[PEER]] == [
        start,
        start + timedelta(minutes=5),
        start + timedelta(minutes=10),
    ]


async def test_deliver_then_fail_records_the_message_and_raises_once() -> None:
    gateway = FakeGateway()
    gateway.deliver_then_fail = TgSendUncertain("ConnectionError")

    with pytest.raises(TgSendUncertain):
        await gateway.send_text(PEER, "Здравствуйте")
    assert gateway.sent == [(PEER, "Здравствуйте")]  # it reached the chat all the same
    assert [m.text for m in gateway.chats[PEER]] == ["Здравствуйте"]
    assert gateway.deliver_then_fail is None

    await gateway.send_text(PEER, "Ещё раз")  # once only
    assert [m.text for m in gateway.chats[PEER]] == ["Здравствуйте", "Ещё раз"]

    gateway.deliver_then_fail = TgSendUncertain("TimeoutError")
    with pytest.raises(TgSendUncertain):
        await gateway.send_file(PEER, "photo.jpg")
    assert gateway.files == [(PEER, "photo.jpg")]
    assert gateway.chats[PEER][-1].text == ""


async def test_recent_messages_reads_after_an_id_and_keeps_the_newest() -> None:
    gateway = FakeGateway()  # the chat is not allowed: it is read all the same
    ids = []
    for n in range(5):
        if n % 2:
            ids.append(await gateway.send_text(PEER, f"№{n}"))
        else:
            ids.append(gateway.add_incoming(PEER, f"№{n}").tg_msg_id)
    gateway.add_incoming(PEER + 1, "другой чат")
    calls_before = list(gateway.calls)

    after_first = await gateway.recent_messages(PEER, ids[0], 10)
    assert [m.text for m in after_first] == ["№1", "№2", "№3", "№4"]  # oldest first
    assert [m.out for m in after_first] == [True, False, True, False]

    newest_two = await gateway.recent_messages(PEER, 0, 2)
    assert [m.tg_msg_id for m in newest_two] == ids[3:]

    assert await gateway.recent_messages(PEER, ids[-1], 10) == []
    assert await gateway.recent_messages(PEER + 2, 0, 10) == []  # nothing there
    assert gateway.calls == [*calls_before, *["recent_messages"] * 4]
    assert gateway.read == []  # nothing is marked read


async def test_recent_messages_fails_when_told_to_once() -> None:
    gateway = FakeGateway()
    gateway.add_incoming(PEER, "Привет")
    gateway.fail_next = TgActionError(TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT_30", 30)

    with pytest.raises(TgActionError):
        await gateway.recent_messages(PEER, 0, 10)
    assert [m.text for m in await gateway.recent_messages(PEER, 0, 10)] == ["Привет"]


# -- the work account's own contacts (Task 5.4, ruling R60) --


def test_a_name_matches_by_its_words_and_by_a_case_ending() -> None:
    assert name_matches("Алишер", "Алишер Каримов")
    assert name_matches("алишера", "Алишер Каримов")  # «узнай у Алишера»
    assert name_matches("Алишер Каримов", "Каримов Алишер")
    assert name_matches("alisher", "Алишер", "alisher_k")
    assert name_matches("ёлкин", "Елкин Пётр")
    assert not name_matches("Алишер", "Алина Каримова")
    assert not name_matches("Алишер Бобур", "Алишер Каримов")  # every word must match
    assert not name_matches("", "Алишер")
    assert not name_matches("Ал", "Алишер")  # too short to tell anyone apart


async def test_search_contacts_finds_the_accounts_own_contacts_by_name() -> None:
    gateway = FakeGateway()
    alisher = PeerInfo(7, "alisher_k", "Алишер Каримов", False, "+998901231234")
    other = PeerInfo(8, None, "Алишер Бобуров", False, "+998935550011")
    gateway.contacts = [alisher, other, PeerInfo(9, "dilshod", "Дилшод", False)]

    assert await gateway.search_contacts("Алишер", 5) == [alisher, other]
    assert await gateway.search_contacts("Алишера Каримова", 5) == [alisher]
    assert await gateway.search_contacts("Алишер", 1) == [alisher]
    assert await gateway.search_contacts("Сардор", 5) == []
    assert gateway.contact_searches == [
        "Алишер", "Алишера Каримова", "Алишер", "Сардор",
    ]  # fmt: skip
    assert gateway.calls == ["search_contacts"] * 4
    assert gateway.searches == []  # never the global search


async def test_search_contacts_fails_when_told_to_once() -> None:
    gateway = FakeGateway()
    gateway.fail_next = TgActionError(TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT_30", 30)

    with pytest.raises(TgActionError):
        await gateway.search_contacts("Алишер", 5)
    assert await gateway.search_contacts("Алишер", 5) == []


async def test_own_username_is_what_it_was_told() -> None:
    gateway = FakeGateway()
    assert await gateway.own_username() is None
    gateway.username = "work_account"
    assert await gateway.own_username() == "work_account"
