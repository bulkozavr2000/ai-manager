"""The owner's control bot on aiogram: shows cards, takes commands, buttons and files.

Only private chats with the owner's ids reach the handlers. Texts go out as plain text (no
parse mode), so nothing a counterpart wrote can turn into markup. In the owner's quiet hours
messages still go out, but silently (no sound or vibration).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from functools import wraps
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    KeyboardButtonRequestUsers,
    Message,
    MessageEntity,
    ReplyKeyboardMarkup,
)

from tg_agent.clock import Clock
from tg_agent.config import Schedule
from tg_agent.control.base import Card, CardGone
from tg_agent.control.chatlog import ChatLog
from tg_agent.control.logic import ControlLogic
from tg_agent.policy.tg_safety import in_quiet_hours

TELEGRAM_LIMIT = 4096
# A button that opens Telegram's own list of people (`cards.PICKER_DATA`).
_USERS = "users:"
# The «/» menu: only the commands that are left (ruling R64).
MENU = (("stop", "Остановить всё"), ("help", "Как со мной говорить"))
# What the owner gets when a handler breaks unexpectedly: never silence.
BROKEN = "Что-то сломалось ({error}). Попробуй ещё раз"

_log = logging.getLogger(__name__)
_SAFE_NAME = re.compile(r"[^\w.\-]+")


def _trim(text: str) -> str:
    return text if len(text) <= TELEGRAM_LIMIT else text[: TELEGRAM_LIMIT - 1] + "…"


def _picker(card: Card) -> ReplyKeyboardMarkup | None:
    """The reply keyboard of a card that offers Telegram's own list of people (`users:<id>`
    buttons, Task 5.4): one tap opens the list, and the person picked comes back as a
    `users_shared` message. Only a sent message can carry it."""
    rows = [
        [
            KeyboardButton(
                text=label,
                request_users=KeyboardButtonRequestUsers(
                    request_id=int(data.removeprefix(_USERS)),
                    user_is_bot=False,
                    max_quantity=1,
                    request_name=True,
                    request_username=True,
                ),
            )
            for label, data in row
            if data.startswith(_USERS)
        ]
        for row in card.buttons
    ]
    rows = [row for row in rows if row]
    if not rows:
        return None
    return ReplyKeyboardMarkup(
        keyboard=rows, resize_keyboard=True, one_time_keyboard=True
    )


def _markup(card: Card) -> InlineKeyboardMarkup | None:
    if not card.buttons or any(
        data.startswith(_USERS) for row in card.buttons for _, data in row
    ):
        return None
    rows = [
        [
            InlineKeyboardButton(text=label, url=data[4:])
            if data.startswith("url:")
            else InlineKeyboardButton(text=label, callback_data=data)
            for label, data in row
        ]
        for row in card.buttons
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


# What the bot does not read (ruling R63), as the owner's chat log names each.
_OTHER_KINDS = {
    "voice": "голосовое сообщение",
    "video": "видео",
    "video_note": "видеосообщение",
    "audio": "аудио",
    "sticker": "стикер",
    "animation": "GIF",
    "location": "место на карте",
    "poll": "опрос",
}
# A contact card is not among them: it gives a person (Task 5.4), as does Telegram's list.
_NOT_TEXT = (
    F.voice
    | F.video
    | F.video_note
    | F.audio
    | F.sticker
    | F.animation
    | F.location
    | F.poll
)


def _person_of(message: Any) -> tuple[str, str | None, str | None]:
    """(name, username, phone) of a contact card or of the first person picked in
    Telegram's list; a field Telegram did not give is None."""
    if message.contact is not None:
        card = message.contact
        name = " ".join(p for p in (card.first_name, card.last_name) if p)
        return name, None, card.phone_number or None
    picked = message.users_shared.users[0] if message.users_shared.users else None
    if picked is None:
        return "", None, None
    name = " ".join(p for p in (picked.first_name, picked.last_name) if p)
    return name, picked.username or None, None


def _forwarded(message: Any) -> bool:
    """Someone else's message the owner forwarded: data, not his words (review I-1)."""
    return (
        getattr(message, "forward_origin", None) is not None
        or getattr(message, "forward_date", None) is not None
    )


def _replies_to(message: Any) -> int | None:
    """The id of the message this one replies to, or None."""
    reply = message.reply_to_message
    return reply.message_id if reply is not None else None


def _what(message: Any) -> str:
    for field, name in _OTHER_KINDS.items():
        if getattr(message, field, None) is not None:
            return name
    return "сообщение"


def _utf16(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _entities(card: Card, text: str) -> list[MessageEntity] | None:
    """The card's bold ranges as Telegram entities, which count UTF-16 code units."""
    entities = [
        MessageEntity(
            type="bold", offset=_utf16(text[:start]), length=_utf16(text[start:end])
        )
        for start, end in ((s, min(e, len(text))) for s, e in card.bold)
        if start < end
    ]
    return entities or None


class BotUI:
    """Implements `UI` for the owner's private chat with the bot."""

    def __init__(
        self, bot: Any, owner_chat_id: int, clock: Clock, schedule: Schedule
    ) -> None:
        self._bot = bot
        self._chat = owner_chat_id
        self._clock = clock
        self._schedule = schedule

    def _silent(self) -> bool:
        return in_quiet_hours(self._clock.now(), self._schedule)

    async def send(self, card: Card, *, silent: bool = False) -> int:
        text = _trim(card.text)
        message = await self._bot.send_message(
            self._chat,
            text,
            reply_markup=_picker(card) or _markup(card),
            entities=_entities(card, text),
            disable_notification=silent or self._silent(),
            parse_mode=None,
        )
        return message.message_id

    async def edit(self, msg_id: int, card: Card) -> None:
        text = _trim(card.text)
        try:
            await self._bot.edit_message_text(
                text,
                chat_id=self._chat,
                message_id=msg_id,
                reply_markup=_markup(card),
                entities=_entities(card, text),
                parse_mode=None,
            )
        except TelegramBadRequest as error:
            if "not modified" in str(error).lower():
                return  # the same text is shown already
            raise CardGone(str(error)) from error

    async def delete(self, msg_id: int) -> None:
        try:
            await self._bot.delete_message(self._chat, msg_id)
        except TelegramBadRequest as error:  # older than 48 hours, or gone already
            raise CardGone(str(error)) from error

    async def pin(self, msg_id: int) -> None:
        await self._bot.pin_chat_message(self._chat, msg_id, disable_notification=True)

    async def unpin(self, msg_id: int) -> None:
        await self._bot.unpin_chat_message(self._chat, message_id=msg_id)

    async def typing(self) -> None:
        await self._bot.send_chat_action(self._chat, "typing")

    async def notify(self, text: str) -> int:
        message = await self._bot.send_message(
            self._chat,
            _trim(text),
            disable_notification=self._silent(),
            parse_mode=None,
        )
        return message.message_id

    async def flush(self) -> int:
        """Nothing is held back: quiet hours make messages silent instead."""
        return 0


async def _save_file(bot: Bot, message: Message, files_dir: Path) -> Path:
    if message.document is not None:
        source, name = message.document, message.document.file_name or "file"
    else:
        source, name = message.photo[-1], "photo.jpg"
    target = files_dir / "inbox" / f"{message.message_id}_{_SAFE_NAME.sub('_', name)}"
    target.parent.mkdir(parents=True, exist_ok=True)
    await bot.download(source, destination=target)
    return target


def build_dispatcher(
    logic: ControlLogic, files_dir: Path, chatlog: ChatLog | None = None
) -> Dispatcher:
    """The bot's handlers; everything that is not the owner's private chat is ignored.

    With a `chatlog`, each of the owner's messages is written down before the logic sees it.
    """
    owners = set(logic.settings.owner.control_user_ids)
    router = Router(name="owner")
    router.message.filter(F.chat.type == ChatType.PRIVATE, F.from_user.id.in_(owners))
    router.callback_query.filter(F.from_user.id.in_(owners))
    router.edited_message.filter(
        F.chat.type == ChatType.PRIVATE, F.from_user.id.in_(owners)
    )

    async def show(cards: list[Card]) -> None:
        for card in cards:
            await logic.show(card)

    def owner_hears_of_errors(
        handler: Callable[..., Awaitable[None]],
    ) -> Callable[..., Awaitable[None]]:
        """An error that escapes a message handler is logged with its traceback, and the
        owner gets one line about it."""

        @wraps(handler)
        async def guarded(message: Message, *args: Any, **kwargs: Any) -> None:
            # His message is the chat's newest; once handled, the layout puts the chat
            # right (stage 5, R52). `settle` never raises.
            await logic.layout.saw(message.message_id)
            try:
                await handler(message, *args, **kwargs)
            except Exception as error:  # noqa: BLE001 - the owner must hear of it
                _log.exception("the bot's %s handler failed", handler.__name__)
                try:
                    await logic.show(Card(BROKEN.format(error=type(error).__name__)))
                except Exception:  # noqa: BLE001 - nothing more can be done
                    _log.exception("could not tell the owner about the failure")
            await logic.layout.settle()

        return guarded

    # A contact card, or a person picked in Telegram's list (the button «Выбрать из
    # контактов»): he chose the person himself (Task 5.4).
    @router.message(F.contact | F.users_shared)
    @owner_hears_of_errors
    async def on_person(message: Message) -> None:
        name, username, phone = _person_of(message)
        if chatlog is not None:
            contact = f"@{username}" if username else phone
            await chatlog.incoming(
                "contact",
                ", ".join(part for part in (name, contact) if part) or "контакт",
                tg_msg_id=message.message_id,
                reply_to_msg_id=_replies_to(message),
            )
        await show(
            await logic.on_person(
                message.from_user.id, name, username, phone, _replies_to(message)
            )
        )

    # Before the files: Telegram sends a GIF with a `document` too.
    @router.message(_NOT_TEXT)
    @owner_hears_of_errors
    async def on_other(message: Message) -> None:
        if chatlog is not None:
            await chatlog.incoming(
                "other",
                _what(message),
                tg_msg_id=message.message_id,
                reply_to_msg_id=_replies_to(message),
            )
        await show(await logic.on_other(message.from_user.id))

    @router.message(F.document | F.photo)
    @owner_hears_of_errors
    async def on_file(message: Message, bot: Bot) -> None:
        if chatlog is not None:
            name = message.document.file_name if message.document else "photo.jpg"
            await chatlog.incoming(
                "file",
                message.caption or name or "file",
                tg_msg_id=message.message_id,
                reply_to_msg_id=_replies_to(message),
            )
        path = await _save_file(bot, message, files_dir)
        reply = message.reply_to_message
        await show(
            await logic.on_file(
                message.from_user.id,
                str(path),
                message.caption or "",
                reply.message_id if reply is not None else None,
                forwarded=_forwarded(message),
            )
        )

    @router.message(F.text.startswith("/"))
    @owner_hears_of_errors
    async def on_command(message: Message) -> None:
        if chatlog is not None:
            await chatlog.incoming(
                "command",
                message.text,
                tg_msg_id=message.message_id,
                reply_to_msg_id=_replies_to(message),
            )
        name, _, args = message.text[1:].partition(" ")
        await show(await logic.on_command(message.from_user.id, name, args.strip()))

    @router.message(F.text)
    @owner_hears_of_errors
    async def on_text(message: Message) -> None:
        reply = message.reply_to_message
        reply_to = reply.message_id if reply is not None else None
        if chatlog is not None:
            await chatlog.incoming(
                "text",
                message.text,
                tg_msg_id=message.message_id,
                reply_to_msg_id=reply_to,
            )
        await show(
            await logic.on_text(
                message.from_user.id,
                message.text,
                reply_to,
                [],
                forwarded=_forwarded(message),
            )
        )

    @router.edited_message(F.text | F.caption)
    @owner_hears_of_errors
    async def on_edit(message: Message) -> None:
        await show(await logic.on_edit(message.from_user.id))

    @router.callback_query()
    async def on_button(query: CallbackQuery) -> None:
        if chatlog is not None:
            card = query.message
            await chatlog.incoming(
                "button",
                query.data or "",
                tg_msg_id=card.message_id if card is not None else None,
            )
        try:
            toast = await logic.on_callback(query.from_user.id, query.data or "")
        except Exception as error:  # noqa: BLE001 - the button must stop spinning
            _log.exception("the bot's button handler failed")
            toast = BROKEN.format(error=type(error).__name__)
        try:
            await query.answer(toast[:200] or None)
        except Exception:  # noqa: BLE001 - too late for Telegram: the chat is put right still
            _log.warning("the button's answer was refused", exc_info=True)
        # A card it answered settles; the bottom of the chat is put right.
        await logic.layout.settle()

    async def set_menu(bot: Bot) -> None:
        """At the bot's start: the commands that no longer exist leave the «/» menu."""
        try:
            await bot.set_my_commands(
                [BotCommand(command=c, description=d) for c, d in MENU]
            )
        except Exception as error:  # noqa: BLE001 - only the menu; the bot works without it
            _log.warning("the bot's menu was not set: %s", error)

    # Telegram's own line «… закрепил(а) сообщение», written under the status each time it is
    # sent anew and pinned (seen live at the stage-5 switch). It comes from the bot itself,
    # so the owner's router never sees it. Deleted: the status stays the chat's last message
    # (ruling R52).
    service = Router(name="service")

    @service.message(
        F.pinned_message, F.chat.type == ChatType.PRIVATE, F.chat.id.in_(owners)
    )
    async def on_pin_line(message: Message) -> None:
        try:
            await message.bot.delete_message(message.chat.id, message.message_id)  # type: ignore[union-attr]
        except Exception as error:  # noqa: BLE001 - only a grey line stays on screen
            _log.info("the pin's line stays: %s", error)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    dispatcher.include_router(service)
    dispatcher.startup.register(set_menu)
    return dispatcher
