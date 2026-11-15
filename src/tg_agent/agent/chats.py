"""The chats of the agent's people in the work account: no sound, and one folder (rulings
R105, R106).

`first_sent` runs when the agent's first message of a task has gone to a person (the `first`
of `Runner.on_sent`). A chat that did not exist before is muted for good. The chat existed
when that message was no first contact (`tasks/first_contacts.is_first_contact`: he wrote
first, or the account wrote to him in another task) or when Telegram holds a message of the
chat older than that one (the owner wrote by hand before the bot), and then it is muted only
if it was not muted already, and the program remembers that it did (kv
`chat_muted:<contact>`); a chat the owner muted himself is left alone. The person also joins
the folder `Settings.chat_folder`.

`task_ended` runs when the owner ends a task (he closed it or pressed «Завершить» under its
result, ruling R109; the agent's result is no end): every chat the program muted for one of
its people gets its sound back, unless another task in work still talks to him
(`Repo.blocking_conversation`). New chats stay muted.

Telegram may refuse any of it (a flood wait, a full folder list, a Premium-only limit): that is
logged once, in plain words, and the send or the end of the task goes on. Only a refused
unmute is tried again, at the next start (`give_back_left_over`).
"""

from __future__ import annotations

import json
import logging

from tg_agent.config import Settings
from tg_agent.db import FIRST_SENT, Repo
from tg_agent.gateway.base import FOLDER_LIMIT, Gateway, TgActionError
from tg_agent.models import Conversation, Direction

_log = logging.getLogger(__name__)

# The chats the program put into the folder, oldest first (a JSON list of peer ids): only
# these may leave it when it is full.
FOLDER_ORDER_KEY = "chat_folder_order"

# How many of the chat's newest messages are read to see whether it existed before our first
# message: more than the agent ever sends at once.
_HISTORY_LOOK = 10


MUTED_PREFIX = "chat_muted:"


def muted_key(contact_id: int) -> str:
    """kv: "1" while the program keeps an existing chat with this person muted."""
    return f"{MUTED_PREFIX}{contact_id}"


def _why(error: Exception) -> str:
    """What went wrong, never a word of any text: a Telegram code or a class name."""
    return str(error) if isinstance(error, TgActionError) else type(error).__name__


async def first_sent(
    repo: Repo, gateway: Gateway, settings: Settings, conv: Conversation
) -> None:
    """The agent's first message of a task went to this person: mute his chat and put it
    into the folder, as the settings say."""
    contact = await repo.get_contact(conv.contact_id)
    if settings.mute_chats:
        await _mute(repo, gateway, conv, contact.id, contact.peer_id)
    if settings.chat_folder:
        await into_folder(repo, gateway, settings.chat_folder, contact.peer_id)


async def _mute(
    repo: Repo, gateway: Gateway, conv: Conversation, contact_id: int, peer_id: int
) -> None:
    # `mark_sent` logged FIRST_SENT for this conversation if the message was a first contact.
    existed = not await repo.events_of(FIRST_SENT, conv.id)
    try:
        if not existed:  # the database may not know a chat the owner had by hand
            existed = await _older_history(repo, gateway, conv.id, peer_id)
        if existed and await gateway.is_muted(peer_id):
            return  # the owner muted it himself, or the program did for another task
        await gateway.set_muted(peer_id, True)
    except Exception as error:  # noqa: BLE001 - Telegram never stops a send
        _log.warning("could not mute the chat of contact %d: %s", contact_id, _why(error))
        return
    if existed:
        await repo.set_kv(muted_key(contact_id), "1")


async def _older_history(
    repo: Repo, gateway: Gateway, conv_id: int, peer_id: int
) -> bool:
    """Does Telegram hold a message of this chat, either way, older than our first message of
    this task? Then the chat existed. Our own messages of this send are newer than the first,
    so two messages sent at once do not count."""
    ours = [
        message.tg_msg_id
        for message in await repo.last_messages(conv_id, _HISTORY_LOOK)
        if message.direction is Direction.OUT and message.tg_msg_id is not None
    ]
    if not ours:
        return False  # its Telegram id is not known: nothing to compare with
    first = min(ours)
    history = await gateway.recent_messages(peer_id, 0, _HISTORY_LOOK)
    return any(message.tg_msg_id < first for message in history)


async def into_folder(repo: Repo, gateway: Gateway, title: str, peer_id: int) -> None:
    """Put the person into the folder `title`. When it is full, the chats the program put
    there longest ago leave it; what the owner put there himself never does."""
    try:
        order: list[int] = json.loads(await repo.get_kv(FOLDER_ORDER_KEY) or "[]")
        peers = await gateway.folder_peers(title)
        if peers is not None and peer_id in peers:
            return
        present = set(peers or ())
        ours = [peer for peer in order if peer in present]  # the owner took the rest out
        over = len(present) + 1 - FOLDER_LIMIT
        remove = ours[: max(over, 0)]
        if len(remove) < over:
            _log.warning("the folder is full of the owner's own chats: a chat not put in")
            return
        await gateway.add_to_folder(title, peer_id, remove)
    except Exception as error:  # noqa: BLE001 - Telegram never stops a send
        _log.warning("could not put a chat into the folder: %s", _why(error))
        return
    kept = [peer for peer in ours if peer not in remove]
    await repo.set_kv(FOLDER_ORDER_KEY, json.dumps([*kept, peer_id]))


async def task_ended(repo: Repo, gateway: Gateway, task_id: int) -> None:
    """The task is over: the chats the program muted for its people get their sound back,
    but not one another task in work still talks in."""
    for conv in await repo.task_conversations(task_id):
        key = muted_key(conv.contact_id)
        if await repo.get_kv(key) != "1":
            continue
        contact = await repo.get_contact(conv.contact_id)
        busy = await repo.blocking_conversation(contact.peer_id)
        if busy is not None and busy.task_id != task_id:
            continue  # that task gives it back when it ends
        try:
            await gateway.set_muted(contact.peer_id, False)
        except Exception as error:  # noqa: BLE001 - Telegram never stops the end of a task
            _log.warning(
                "could not give the sound back to the chat of contact %d: %s",
                contact.id,
                _why(error),
            )
            continue
        await repo.set_kv(key, "")


async def give_back_left_over(repo: Repo, gateway: Gateway) -> None:
    """At start-up: a sound Telegram refused to give back at a task's end is tried again for
    every person no task in work talks to (that task gives it back when it ends)."""
    in_work = {contact.id for contact in await repo.contacts_in_work()}
    for key in await repo.kv_keys(MUTED_PREFIX):
        contact_id = int(key.removeprefix(MUTED_PREFIX))
        if await repo.get_kv(key) != "1" or contact_id in in_work:
            continue
        contact = await repo.get_contact(contact_id)
        try:
            await gateway.set_muted(contact.peer_id, False)
        except Exception as error:  # noqa: BLE001 - Telegram never stops the start
            _log.warning(
                "could not give the sound back to the chat of contact %d: %s",
                contact.id,
                _why(error),
            )
            continue
        await repo.set_kv(key, "")
