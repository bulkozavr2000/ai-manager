"""A Telegram double for tests: it records what was done and fails when told to."""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime

from tg_agent.clock import Clock
from tg_agent.gateway.base import (
    FOLDER_LIMIT,
    ChatMessage,
    PeerInfo,
    PublicPeer,
    TgActionError,
    TgSendUncertain,
    name_matches,
)
from tg_agent.policy.tg_safety import TgErrorKind


class FakeGateway:
    """Implements `Gateway` in memory.

    `calls` lists the method names in call order. `fail_next`, when set, is raised (once) by the
    next `resolve`, `send_text` or `send_file`, before that call does anything: a
    `TgActionError`, or a `TgSendUncertain` for a send that may or may not have gone out (here
    it did not). `fail_typing`,
    when set, is raised by every `typing` call (after it is recorded in `calls`, and without
    recording it in `typed`) until it is set back to None: any exception, `TgActionError` or
    not. `peers` maps a username (without "@") to what `resolve` returns; `media` maps
    (peer_id, tg_msg_id) to what `download` returns. An unknown username or message raises
    `TgActionError` (OTHER). `replay` records the requests of an allowed chat in `replays` and
    answers `replay_count`; for a chat that is not allowed it answers 0 and records nothing;
    `fail_next` makes it raise like the other reads.

    `chats` is a log per peer: `send_text` and `send_file` append an outgoing entry (a file's
    text is ""), `add_incoming` scripts a counterpart's message, and `recent_messages` reads
    it, of an allowed chat or not (`fail_next` makes it raise like the other reads).
    `deliver_then_fail`, when set, makes the next `send_text` or `send_file` record the
    message as delivered (in `sent` / `files` and in `chats`) and then raise it, once: the
    send broke off, but the message reached the chat.

    With a `clock`, the chat log is dated by it (so "newer than the attempt" follows the test's
    time); without one, by the wall clock.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock
        self.sent: list[tuple[int, str]] = []  # (peer_id, text) of every send_text
        self.files: list[tuple[int, str]] = []  # (peer_id, path) of every send_file
        self.calls: list[str] = []
        self.fail_next: TgActionError | TgSendUncertain | None = None
        self.fail_typing: BaseException | None = None
        self.peers: dict[str, PeerInfo] = {}
        self.media: dict[tuple[int, int], bytes] = {}
        self.typed: list[tuple[int, float]] = []
        self.read: list[tuple[int, int]] = []
        self.allowed: set[int] = set()
        self.replays: list[
            tuple[int, int, set[int]]
        ] = []  # (peer_id, after, known) asked
        self.replay_count = 0  # what replay answers for an allowed chat
        self.name = ""  # the account's own name (own_name)
        # What search_public finds: every entry whose username, title or about holds a word
        # of the query. `searches` records the queries.
        self.public: list[PublicPeer] = []
        self.searches: list[str] = []
        # The account's address book, for search_contacts (`name_matches` on the name and
        # the username); `contact_searches` records the queries.
        self.contacts: list[PeerInfo] = []
        self.contact_searches: list[str] = []
        self.username: str | None = None  # the account's own username (own_username)
        self.chats: dict[int, list[ChatMessage]] = {}
        self.deliver_then_fail: BaseException | None = None
        self._last_msg_id = 0
        # The chats' sound and the folders (R105, R106): `muted` per peer (unknown: not
        # muted), `mute_calls` the (peer, muted) of every `set_muted`, `folders` the chats of
        # each folder by title, `folders_made` how many were made. `fail_chats`, while set, is
        # raised by each of these four methods, before it does anything.
        self.muted: dict[int, bool] = {}
        self.mute_calls: list[tuple[int, bool]] = []
        self.folders: dict[str, list[int]] = {}
        self.folders_made = 0
        self.fail_chats: BaseException | None = None

    def _fail_if_told(self) -> None:
        failure, self.fail_next = self.fail_next, None
        if failure is not None:
            raise failure

    def _new_msg_id(self) -> int:
        self._last_msg_id += 1
        return self._last_msg_id

    def _log(self, peer_id: int, out: bool, text: str) -> ChatMessage:
        date = self._clock.now() if self._clock is not None else datetime.now(UTC)
        message = ChatMessage(self._new_msg_id(), out, text, date)
        self.chats.setdefault(peer_id, []).append(message)
        return message

    def _delivered(self, peer_id: int, text: str) -> int:
        """Write an outgoing message into the chat; raise `deliver_then_fail` if it is set."""
        message = self._log(peer_id, True, text)
        failure, self.deliver_then_fail = self.deliver_then_fail, None
        if failure is not None:
            raise failure
        return message.tg_msg_id

    def add_incoming(self, peer_id: int, text: str) -> ChatMessage:
        """Script a counterpart's message into the chat log (no callback is called)."""
        return self._log(peer_id, False, text)

    async def resolve(self, target: str) -> PeerInfo:
        """`peers` is keyed by username (without "@") or by phone ("+998...")."""
        self.calls.append("resolve")
        self._fail_if_told()
        wanted = target.removeprefix("@").casefold()
        for username, peer in self.peers.items():
            if username.casefold() == wanted:
                return peer
        raise TgActionError(TgErrorKind.OTHER, f"no user {target!r}")

    async def search_public(self, query: str, limit: int) -> list[PublicPeer]:
        self.calls.append("search_public")
        self._fail_if_told()
        self.searches.append(query)
        words = [word.casefold() for word in query.split() if word]
        found = [
            peer
            for peer in self.public
            if any(
                word in f"{peer.username} {peer.title} {peer.about}".casefold()
                for word in words
            )
        ]
        return found[:limit]

    async def search_contacts(self, query: str, limit: int) -> list[PeerInfo]:
        self.calls.append("search_contacts")
        self._fail_if_told()
        self.contact_searches.append(query)
        found = [
            peer
            for peer in self.contacts
            if not peer.is_bot
            and name_matches(query, peer.display_name, peer.username or "")
        ]
        return found[:limit]

    async def own_name(self) -> str:
        self._fail_if_told()
        return self.name

    async def own_username(self) -> str | None:
        return self.username

    async def send_text(self, peer_id: int, text: str) -> int:
        self.calls.append("send_text")
        self._fail_if_told()
        self.sent.append((peer_id, text))
        return self._delivered(peer_id, text)

    async def send_file(self, peer_id: int, path: str) -> int:
        self.calls.append("send_file")
        self._fail_if_told()
        self.files.append((peer_id, path))
        return self._delivered(peer_id, "")

    async def typing(self, peer_id: int, seconds: float) -> None:
        self.calls.append("typing")
        if self.fail_typing is not None:
            raise self.fail_typing
        self.typed.append((peer_id, seconds))

    async def mark_read(self, peer_id: int, max_id: int) -> None:
        self.calls.append("mark_read")
        self.read.append((peer_id, max_id))

    async def download(self, peer_id: int, tg_msg_id: int) -> bytes:
        self.calls.append("download")
        try:
            return self.media[(peer_id, tg_msg_id)]
        except KeyError:
            raise TgActionError(
                TgErrorKind.OTHER, f"no media in message {tg_msg_id}"
            ) from None

    async def set_allowed(self, peer_ids: set[int]) -> None:
        self.calls.append("set_allowed")
        self.allowed = set(peer_ids)

    async def replay(
        self, peer_id: int, after_msg_id: int, known_out_ids: set[int]
    ) -> int:
        self.calls.append("replay")
        self._fail_if_told()
        if peer_id not in self.allowed:
            return 0
        self.replays.append((peer_id, after_msg_id, set(known_out_ids)))
        return self.replay_count

    async def recent_messages(
        self, peer_id: int, after_msg_id: int, limit: int
    ) -> list[ChatMessage]:
        self.calls.append("recent_messages")
        self._fail_if_told()
        newer = [m for m in self.chats.get(peer_id, []) if m.tg_msg_id > after_msg_id]
        return newer[-limit:] if limit > 0 else []

    # -- the chats' sound and the folder: never in `calls`, never `fail_next` --

    def _fail_chats_if_told(self) -> None:
        if self.fail_chats is not None:
            raise self.fail_chats

    async def is_muted(self, peer_id: int) -> bool:
        self._fail_chats_if_told()
        return self.muted.get(peer_id, False)

    async def set_muted(self, peer_id: int, muted: bool) -> None:
        self._fail_chats_if_told()
        self.mute_calls.append((peer_id, muted))
        self.muted[peer_id] = muted

    async def folder_peers(self, title: str) -> list[int] | None:
        self._fail_chats_if_told()
        found = self.folders.get(title)
        return None if found is None else list(found)

    async def add_to_folder(
        self, title: str, peer_id: int, remove: Collection[int] = ()
    ) -> None:
        self._fail_chats_if_told()
        if title not in self.folders:
            self.folders_made += 1
        kept = [peer for peer in self.folders.get(title, []) if peer not in remove]
        if peer_id not in kept:
            kept.append(peer_id)
        if len(kept) > FOLDER_LIMIT:  # Telegram refuses it; nothing changes
            raise TgActionError(TgErrorKind.OTHER, "folder full")
        self.folders[title] = kept
