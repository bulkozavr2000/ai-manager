"""The chats of the agent's people in the work account (`agent/chats.py`, rulings R105, R106):
muted while the agent talks to them, and gathered in the folder «ИИ агент». On fakes."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import replace

import pytest

from tg_agent.agent import chats, people
from tg_agent.agent.ops import TaskOps
from tg_agent.agent.people import save_targets
from tg_agent.agent.runner import Runner
from tg_agent.brain.llm import FakeLLM
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.gateway.base import FOLDER_LIMIT, PeerInfo, TgActionError
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import TaskStatus
from tg_agent.policy.tg_safety import TgErrorKind
from tg_agent.tasks.outbox import mark_sent

FOLDER = "ИИ агент"


class Pieces:
    """The runner and the owner's task actions over fakes."""

    def __init__(self, repo: Repo, clock: FakeClock, settings: Settings) -> None:
        self.repo, self.clock = repo, clock
        self.gateway = FakeGateway(clock)
        self.use(settings)
        self._peer = 700

    def use(self, settings: Settings) -> None:
        self.settings = settings
        self.runner = Runner(
            self.repo, FakeLLM(), self.gateway, FakeUI(), self.clock, settings,
            random.Random(3), canary="ZX-CHATS-1", owner_secrets=[], inline=True,
        )  # fmt: skip
        self.ops = TaskOps(
            self.repo, self.gateway, self.clock, settings, self.runner, random.Random(5)
        )

    def peer(self, username: str) -> int:
        if username in self.gateway.peers:
            return self.gateway.peers[username].peer_id
        self._peer += 1
        self.gateway.peers[username] = PeerInfo(
            self._peer, username, username.title(), False
        )
        return self._peer

    async def task(self, *usernames: str) -> int:
        """A task in work with these people attached."""
        for username in usernames:
            self.peer(username)
        task_id = await self.repo.create_task(
            "Дрель", "Купить дрель", {"what": "дрель"}, None, "UZS", [], [],
            source_text="Найди дрель",
        )  # fmt: skip
        await save_targets(self.repo, task_id, list(usernames))
        launched, notes = await self.ops.launch(task_id)
        assert launched, notes
        return task_id

    async def conv(self, task_id: int, username: str) -> int:
        peer_id = self.peer(username)
        for conv in await self.repo.task_conversations(task_id):
            if (await self.repo.get_contact(conv.contact_id)).peer_id == peer_id:
                return conv.id
        raise AssertionError(username)

    async def first_message(
        self, task_id: int, username: str, *, he_wrote: bool = False
    ) -> None:
        """The agent's first message of the task goes to him, the way the Sender reports it.
        `he_wrote`: he wrote to the account first (the chat exists)."""
        conv_id = await self.conv(task_id, username)
        if he_wrote:
            await self.repo.update_conversation(conv_id, last_in_at=self.clock.now())
        conv = await self.repo.get_conversation(conv_id)
        item_id = await self.repo.enqueue(conv_id, "Здравствуйте!", self.clock.now())
        item = await self.repo.get_outbox(item_id)
        msg_id = await self.gateway.send_text(self.peer(username), item.text)
        first = await mark_sent(
            self.repo, item, conv, msg_id, self.clock.now(), is_file=False
        )
        assert first
        await self.runner.on_sent(item, conv, self.clock.now(), first=first)

    def muted(self, username: str) -> bool:
        return self.gateway.muted.get(self.peer(username), False)

    async def done(self, task_id: int) -> None:
        """The agent reported and the owner pressed «Завершить» (R109)."""
        await self.repo.set_kv(f"task_result_waits:{task_id}", "1")
        assert await self.ops.acknowledge(task_id) == f"Задача #{task_id} завершена"
        assert (await self.repo.get_task(task_id)).status is TaskStatus.DONE


@pytest.fixture
def w(repo, fake_clock, settings) -> Pieces:
    return Pieces(repo, fake_clock, settings)


# -- R105: muted --


def test_chats_are_muted_and_gathered_by_default() -> None:
    assert Settings().mute_chats is True
    assert Settings().chat_folder == FOLDER


async def test_a_new_chat_is_muted_for_good(w: Pieces) -> None:
    task_id = await w.task("new_shop")

    await w.first_message(task_id, "new_shop")
    assert w.muted("new_shop")

    await w.ops.close(task_id)
    assert w.muted("new_shop")  # a new chat stays muted
    assert w.gateway.mute_calls == [(w.peer("new_shop"), True)]


async def test_an_existing_chat_is_muted_while_the_task_is_in_work(w: Pieces) -> None:
    task_id = await w.task("old_friend")

    await w.first_message(task_id, "old_friend", he_wrote=True)
    assert w.muted("old_friend")

    await w.ops.close(task_id)
    assert not w.muted("old_friend")  # not answering him with no sound would be rude
    assert w.gateway.mute_calls == [
        (w.peer("old_friend"), True),
        (w.peer("old_friend"), False),
    ]


async def test_an_existing_chat_gets_its_sound_back_when_the_owner_accepts_the_result(
    w: Pieces,
) -> None:
    task_id = await w.task("old_friend")
    await w.first_message(task_id, "old_friend", he_wrote=True)

    await w.done(task_id)

    assert not w.muted("old_friend")


async def test_a_person_written_to_in_an_earlier_task_has_an_existing_chat(
    w: Pieces,
) -> None:
    first = await w.task("shop")
    await w.first_message(first, "shop")  # new then: muted for good
    await w.ops.close(first)
    w.gateway.muted.clear()  # the owner gave the chat its sound back himself

    second = await w.task("shop")
    await w.first_message(second, "shop")
    assert w.muted("shop")
    await w.ops.close(second)

    assert not w.muted("shop")


async def test_a_chat_with_history_the_database_does_not_know_is_an_existing_chat(
    w: Pieces,
) -> None:
    # Fix round 9.2 (D4): the owner wrote to him by hand before the bot; only Telegram has it.
    peer = w.peer("old_supplier")
    w.gateway._log(peer, True, "Салам, как дела?")
    w.gateway._log(peer, False, "Всё хорошо!")
    task_id = await w.task("old_supplier")

    await w.first_message(task_id, "old_supplier")
    assert w.muted("old_supplier")
    await w.ops.close(task_id)

    assert not w.muted("old_supplier")
    assert w.gateway.mute_calls == [(peer, True), (peer, False)]


async def test_a_new_chat_with_two_own_messages_is_still_new(w: Pieces) -> None:
    # The agent often sends two messages at once: the second is no older history.
    task_id = await w.task("new_shop")
    peer = w.peer("new_shop")
    conv_id = await w.conv(task_id, "new_shop")
    conv = await w.repo.get_conversation(conv_id)
    item = await w.repo.get_outbox(
        await w.repo.enqueue(conv_id, "Здравствуйте!", w.clock.now())
    )
    msg_id = await w.gateway.send_text(peer, item.text)
    await w.gateway.send_text(peer, "Подскажите цену")  # the second one, already out
    first = await mark_sent(w.repo, item, conv, msg_id, w.clock.now(), is_file=False)
    await w.runner.on_sent(item, conv, w.clock.now(), first=first)

    await w.ops.close(task_id)

    assert w.muted("new_shop")
    assert w.gateway.mute_calls == [(peer, True)]


async def test_the_sound_waits_while_another_task_in_work_talks_to_him(
    w: Pieces,
) -> None:
    first = await w.task("old_friend")
    await w.first_message(first, "old_friend", he_wrote=True)
    # The same person in a second task in work (he was let go by the first one meanwhile).
    second = await w.task("other")
    conv = await w.repo.create_conversation(
        second, (await w.repo.contact_by_peer(w.peer("old_friend"))).id
    )
    await w.repo.update_conversation(conv, first_sent=True)

    await w.ops.close(first)
    assert w.muted("old_friend")

    await w.ops.close(second)
    assert not w.muted("old_friend")


async def test_a_chat_the_owner_muted_himself_is_left_alone(w: Pieces) -> None:
    w.gateway.muted[w.peer("old_friend")] = True
    task_id = await w.task("old_friend")

    await w.first_message(task_id, "old_friend", he_wrote=True)
    await w.ops.close(task_id)

    assert w.muted("old_friend")
    assert w.gateway.mute_calls == []


async def test_muting_can_be_switched_off(w: Pieces) -> None:
    w.use(replace(w.settings, mute_chats=False, chat_folder=""))
    task_id = await w.task("new_shop", "old_friend")

    await w.first_message(task_id, "new_shop")
    await w.first_message(task_id, "old_friend", he_wrote=True)
    await w.ops.close(task_id)

    assert w.gateway.mute_calls == []
    assert w.gateway.folders == {}


async def test_only_the_first_message_of_a_task_mutes(w: Pieces) -> None:
    task_id = await w.task("new_shop")
    await w.first_message(task_id, "new_shop")
    w.gateway.muted.clear()  # the owner gave it its sound back during the task
    conv = await w.repo.get_conversation(await w.conv(task_id, "new_shop"))
    item = await w.repo.get_outbox(await w.repo.enqueue(conv.id, "Ещё", w.clock.now()))

    await w.runner.on_sent(item, conv, w.clock.now(), first=False)

    assert not w.muted("new_shop")


# -- R105: Telegram refuses --


async def test_a_refused_mute_never_stops_the_send_and_is_logged_once(
    w: Pieces, caplog: pytest.LogCaptureFixture
) -> None:
    task_id = await w.task("old_friend")
    w.gateway.fail_chats = TgActionError(TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT_30", 30)

    with caplog.at_level(logging.WARNING, logger="tg_agent.agent.chats"):
        await w.first_message(task_id, "old_friend", he_wrote=True)

    mute_lines = [r for r in caplog.records if "mute" in r.getMessage()]
    assert len(mute_lines) == 1
    assert "Здравствуйте" not in caplog.text
    # The send went on.
    conv = await w.repo.get_conversation(await w.conv(task_id, "old_friend"))
    assert conv.first_sent
    # Nothing was muted, so nothing is given back at the end.
    w.gateway.fail_chats = None
    await w.ops.close(task_id)
    assert w.gateway.mute_calls == []


async def test_a_refused_unmute_never_stops_the_end_of_the_task(
    w: Pieces, caplog: pytest.LogCaptureFixture
) -> None:
    task_id = await w.task("old_friend")
    await w.first_message(task_id, "old_friend", he_wrote=True)
    w.gateway.fail_chats = TgActionError(TgErrorKind.OTHER, "PEER_ID_INVALID")

    with caplog.at_level(logging.WARNING, logger="tg_agent.agent.chats"):
        await w.ops.close(task_id)

    assert (await w.repo.get_task(task_id)).status is TaskStatus.CANCELLED
    assert len([r for r in caplog.records if "sound" in r.getMessage()]) == 1
    assert w.muted("old_friend")


async def test_a_refused_unmute_is_tried_again_at_start_up(
    w: Pieces, caplog: pytest.LogCaptureFixture
) -> None:
    # Fix round 9.2 (D5).
    task_id = await w.task("old_friend")
    await w.first_message(task_id, "old_friend", he_wrote=True)
    w.gateway.fail_chats = TgActionError(TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT_300", 300)
    await w.ops.close(task_id)
    contact = await w.repo.contact_by_peer(w.peer("old_friend"))
    assert w.muted("old_friend")

    with caplog.at_level(logging.WARNING, logger="tg_agent.agent.chats"):
        await chats.give_back_left_over(w.repo, w.gateway)  # still refused: kept
    assert w.muted("old_friend")
    assert await w.repo.get_kv(chats.muted_key(contact.id)) == "1"
    assert len([r for r in caplog.records if "sound" in r.getMessage()]) == 2

    w.gateway.fail_chats = None
    await chats.give_back_left_over(w.repo, w.gateway)  # the next start

    assert not w.muted("old_friend")
    assert not await w.repo.get_kv(chats.muted_key(contact.id))


async def test_start_up_leaves_the_mute_of_a_person_of_a_task_in_work(w: Pieces) -> None:
    task_id = await w.task("old_friend")
    await w.first_message(task_id, "old_friend", he_wrote=True)
    contact = await w.repo.contact_by_peer(w.peer("old_friend"))
    await people.close(w.repo, w.gateway, await w.conv(task_id, "old_friend"), "отказался")

    await chats.give_back_left_over(w.repo, w.gateway)

    assert w.muted("old_friend")  # his task is in work: it gives the sound back at its end
    assert await w.repo.get_kv(chats.muted_key(contact.id)) == "1"


async def test_a_gateway_that_breaks_otherwise_never_stops_a_send(
    w: Pieces, caplog: pytest.LogCaptureFixture
) -> None:
    task_id = await w.task("new_shop")
    w.gateway.fail_chats = ConnectionError("the link broke")

    with caplog.at_level(logging.WARNING, logger="tg_agent.agent.chats"):
        await w.first_message(task_id, "new_shop")
        await w.ops.close(task_id)

    assert "ConnectionError" in caplog.text
    assert (await w.repo.get_task(task_id)).status is TaskStatus.CANCELLED


# -- R106: the folder --


async def test_the_folder_is_made_with_the_first_person_and_found_afterwards(
    w: Pieces,
) -> None:
    task_id = await w.task("new_shop", "old_friend")

    await w.first_message(task_id, "new_shop")
    await w.first_message(task_id, "old_friend", he_wrote=True)

    assert w.gateway.folders == {
        FOLDER: [w.peer("new_shop"), w.peer("old_friend")]
    }
    assert w.gateway.folders_made == 1


async def test_a_person_is_put_into_the_folder_once(w: Pieces) -> None:
    first = await w.task("shop")
    await w.first_message(first, "shop")
    await w.ops.close(first)
    second = await w.task("shop")
    await w.first_message(second, "shop")

    assert w.gateway.folders == {FOLDER: [w.peer("shop")]}


async def test_a_full_folder_lets_the_programs_oldest_go_and_keeps_the_owners(
    repo: Repo, fake_clock: FakeClock
) -> None:
    gateway = FakeGateway(fake_clock)
    owners = [1, 2, 3]
    gateway.folders[FOLDER] = list(owners)  # the owner put three chats there himself
    ours = list(range(1001, 1001 + FOLDER_LIMIT - len(owners)))
    for peer in ours:
        await chats.into_folder(repo, gateway, FOLDER, peer)
    assert len(gateway.folders[FOLDER]) == FOLDER_LIMIT

    await chats.into_folder(repo, gateway, FOLDER, 5000)

    folder = gateway.folders[FOLDER]
    assert len(folder) == FOLDER_LIMIT
    assert ours[0] not in folder and 5000 in folder
    assert all(peer in folder for peer in owners)
    assert json.loads(await repo.get_kv(chats.FOLDER_ORDER_KEY)) == [*ours[1:], 5000]


async def test_a_folder_full_of_the_owners_own_chats_is_not_touched(
    repo: Repo, fake_clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    gateway = FakeGateway(fake_clock)
    gateway.folders[FOLDER] = list(range(1, FOLDER_LIMIT + 1))

    with caplog.at_level(logging.WARNING, logger="tg_agent.agent.chats"):
        await chats.into_folder(repo, gateway, FOLDER, 5000)

    assert gateway.folders[FOLDER] == list(range(1, FOLDER_LIMIT + 1))
    assert len(caplog.records) == 1


async def test_what_the_owner_took_out_of_the_folder_is_forgotten(
    repo: Repo, fake_clock: FakeClock
) -> None:
    gateway = FakeGateway(fake_clock)
    await chats.into_folder(repo, gateway, FOLDER, 11)
    await chats.into_folder(repo, gateway, FOLDER, 12)
    gateway.folders[FOLDER].remove(11)  # the owner took him out

    await chats.into_folder(repo, gateway, FOLDER, 13)

    assert json.loads(await repo.get_kv(chats.FOLDER_ORDER_KEY)) == [12, 13]


async def test_a_broken_folder_order_never_stops_a_send(
    repo: Repo, fake_clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    # Fix round 9.2 (list 11): the kv value is read inside the guard.
    gateway = FakeGateway(fake_clock)
    await repo.set_kv(chats.FOLDER_ORDER_KEY, "[1, 2")

    with caplog.at_level(logging.WARNING, logger="tg_agent.agent.chats"):
        await chats.into_folder(repo, gateway, FOLDER, 5000)

    assert len([r for r in caplog.records if "folder" in r.getMessage()]) == 1


async def test_a_refused_folder_never_stops_a_send(
    w: Pieces, caplog: pytest.LogCaptureFixture
) -> None:
    w.use(replace(w.settings, mute_chats=False))
    task_id = await w.task("new_shop")
    w.gateway.fail_chats = TgActionError(TgErrorKind.OTHER, "FILTERS_TOO_MUCH")

    with caplog.at_level(logging.WARNING, logger="tg_agent.agent.chats"):
        await w.first_message(task_id, "new_shop")

    assert len([r for r in caplog.records if "folder" in r.getMessage()]) == 1
    assert w.gateway.folders == {}
    assert await w.repo.get_kv(chats.FOLDER_ORDER_KEY) is None
