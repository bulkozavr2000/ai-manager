from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from aiogram.exceptions import TelegramBadRequest

from tg_agent.clock import FakeClock
from tg_agent.config import Schedule
from tg_agent.control.base import Bold, Card, CardGone, compose
from tg_agent.control.bot import TELEGRAM_LIMIT, BotUI, build_dispatcher
from tg_agent.control.chatlog import ChatLog
from tg_agent.db import Repo
from tg_agent.models import Direction

TZ = ZoneInfo("Asia/Tashkent")


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.edited: list[dict] = []
        self.pinned: list[dict] = []
        self.deleted: list[tuple[int, int]] = []

    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        self.deleted.append((chat_id, message_id))
        return True

    async def send_message(self, chat_id: int, text: str, **kwargs: object) -> object:
        self.sent.append({"chat_id": chat_id, "text": text, **kwargs})
        return SimpleNamespace(message_id=len(self.sent))

    async def edit_message_text(self, text: str, **kwargs: object) -> object:
        self.edited.append({"text": text, **kwargs})
        return True

    async def pin_chat_message(
        self, chat_id: int, message_id: int, **kwargs: object
    ) -> bool:
        self.pinned.append({"chat_id": chat_id, "message_id": message_id, **kwargs})
        return True


def ui_at(hour: int) -> tuple[BotUI, FakeBot]:
    bot = FakeBot()
    clock = FakeClock(datetime(2026, 11, 10, hour, 0, tzinfo=TZ))
    return BotUI(bot, owner_chat_id=111, clock=clock, schedule=Schedule()), bot


async def test_send_builds_callback_and_url_buttons() -> None:
    ui, bot = ui_at(10)
    card = Card(
        "Вопрос",
        [[("Дать", "ap:1:yes:abcdef")], [("Открыть чат", "url:https://t.me/x_shop")]],
    )
    msg_id = await ui.send(card)
    assert msg_id == 1
    (sent,) = bot.sent
    rows = sent["reply_markup"].inline_keyboard
    assert rows[0][0].callback_data == "ap:1:yes:abcdef"
    assert rows[1][0].url == "https://t.me/x_shop" and rows[1][0].callback_data is None
    assert sent["chat_id"] == 111 and sent["disable_notification"] is False


async def test_quiet_hours_send_silently() -> None:
    ui, bot = ui_at(22)
    await ui.send(Card("Ночью"))
    await ui.notify("И это ночью")
    assert [s["disable_notification"] for s in bot.sent] == [True, True]


async def test_text_is_sent_plain_and_trimmed_to_the_telegram_limit() -> None:
    ui, bot = ui_at(10)
    await ui.notify("<b>" + "я" * 5000)
    (sent,) = bot.sent
    assert len(sent["text"]) <= TELEGRAM_LIMIT
    assert sent["text"].startswith("<b>")
    assert sent.get("parse_mode") is None


async def test_edit_replaces_text_and_buttons() -> None:
    ui, bot = ui_at(10)
    await ui.edit(5, Card("Новое", [[("Ок", "ap:1:resume:abcdef")]]))
    (edited,) = bot.edited
    assert (
        edited["message_id"] == 5
        and edited["chat_id"] == 111
        and edited["text"] == "Новое"
    )


async def test_flush_has_nothing_to_send() -> None:
    ui, _ = ui_at(10)
    assert await ui.flush() == 0


async def test_pin_is_silent() -> None:
    ui, bot = ui_at(12)
    await ui.pin(5)
    assert bot.pinned == [
        {"chat_id": 111, "message_id": 5, "disable_notification": True}
    ]


class RefusingBot(FakeBot):
    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason

    async def edit_message_text(self, text: str, **kwargs: object) -> object:
        raise TelegramBadRequest(method=None, message=self.reason)

    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        raise TelegramBadRequest(method=None, message=self.reason)


async def test_a_card_can_be_sent_silently_in_the_day() -> None:
    ui, bot = ui_at(10)
    await ui.send(Card("Статус"), silent=True)
    await ui.send(Card("Вопрос"))
    assert [s["disable_notification"] for s in bot.sent] == [True, False]


async def test_delete_removes_the_message_and_a_refusal_is_card_gone() -> None:
    ui, bot = ui_at(12)
    await ui.delete(7)
    assert bot.deleted == [(111, 7)]
    ui._bot = RefusingBot("Bad Request: message can't be deleted for everyone")
    with pytest.raises(CardGone):
        await ui.delete(8)


async def test_an_edit_telegram_refuses_is_card_gone() -> None:
    ui, _ = ui_at(12)
    ui._bot = RefusingBot("Bad Request: message to edit not found")
    with pytest.raises(CardGone):
        await ui.edit(5, Card("x"))


async def test_an_edit_to_the_same_text_is_no_error() -> None:
    ui, _ = ui_at(12)
    ui._bot = RefusingBot("Bad Request: message is not modified")
    await ui.edit(5, Card("x"))


async def test_bold_pieces_become_entities_counted_in_utf16_units() -> None:
    ui, bot = ui_at(12)
    await ui.send(compose([Bold("🧩 Задача"), "\nтекст ", Bold("жирно")]))
    entities = bot.sent[-1]["entities"]
    assert [(e.type, e.offset, e.length) for e in entities] == [
        ("bold", 0, 9),
        ("bold", 16, 5),
    ]
    assert bot.sent[-1]["parse_mode"] is None


async def test_notify_returns_the_telegram_message_id() -> None:
    ui, bot = ui_at(10)
    await ui.send(Card("Карточка"))
    assert await ui.notify("Готово") == 2
    assert [s["text"] for s in bot.sent] == ["Карточка", "Готово"]


OWNER = 111


class SpyLayout:
    """Stands in for the chat's layout: records what the handlers report and ask."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def saw(self, msg_id: int) -> None:
        self.calls.append(f"saw {msg_id}")

    async def settle(self) -> None:
        self.calls.append("settle")


class SpyLogic:
    """Stands in for ControlLogic: records what the owner's chat log held at each call."""

    def __init__(self, repo: Repo) -> None:
        self.repo = repo
        self.settings = SimpleNamespace(owner=SimpleNamespace(control_user_ids=[OWNER]))
        self.seen: list[tuple[str, list[tuple[str, str]]]] = []
        self.layout = SpyLayout()

    async def _saw(self, name: str) -> list[Card]:
        rows = await self.repo.owner_chat_tail(10)
        self.seen.append((name, [(r.kind, r.text) for r in rows]))
        return []

    async def on_text(self, *args: object, **kwargs: object) -> list[Card]:
        return await self._saw("text")

    async def on_command(self, *args: object) -> list[Card]:
        return await self._saw("command")

    async def on_file(self, *args: object, **kwargs: object) -> list[Card]:
        return await self._saw("file")

    async def on_callback(self, *args: object) -> str:
        await self._saw("button")
        return ""

    async def on_other(self, *args: object) -> list[Card]:
        return await self._saw("other")

    async def on_edit(self, *args: object) -> list[Card]:
        self.edits = [*getattr(self, "edits", []), args]
        return []

    async def show(self, card: Card) -> None:
        pass


def handlers(logic: SpyLogic, tmp_path: Path, chatlog: ChatLog | None) -> dict:
    dispatcher = build_dispatcher(logic, tmp_path, chatlog)  # type: ignore[arg-type]
    # The owner's router; a second one only removes Telegram's line about a pin.
    (router,) = (r for r in dispatcher.sub_routers if r.name == "owner")
    found = router.message.handlers + router.callback_query.handlers
    return {h.callback.__name__: h.callback for h in found}


def message(message_id: int, **fields: object) -> SimpleNamespace:
    base = {
        "message_id": message_id,
        "from_user": SimpleNamespace(id=OWNER),
        "text": None,
        "caption": None,
        "document": None,
        "photo": None,
        "reply_to_message": None,
    }
    return SimpleNamespace(**(base | fields))


class DownloadingBot:
    async def download(self, source: object, destination: Path) -> None:
        destination.write_bytes(b"x")


async def test_each_handler_logs_the_owner_message_before_the_logic(
    repo: Repo, tmp_path: Path
) -> None:
    logic = SpyLogic(repo)
    on = handlers(logic, tmp_path, ChatLog(repo))
    await on["on_text"](
        message(41, text="добавь @x", reply_to_message=SimpleNamespace(message_id=40))
    )
    await on["on_command"](message(42, text="/task 3"))
    document = SimpleNamespace(file_name="макет.stl")
    await on["on_file"](message(43, document=document), DownloadingBot())
    await on["on_file"](
        message(44, photo=[SimpleNamespace()], caption="вот так"), DownloadingBot()
    )

    async def answer(text: str | None) -> None:
        pass

    query = SimpleNamespace(
        from_user=SimpleNamespace(id=OWNER),
        data="tk:1:launch:abcdef",
        message=SimpleNamespace(message_id=40),
        answer=answer,
    )
    await on["on_button"](query)

    # Each call of the logic already found its own message in the log.
    assert [(name, rows[-1]) for name, rows in logic.seen] == [
        ("text", ("text", "добавь @x")),
        ("command", ("command", "/task 3")),
        ("file", ("file", "макет.stl")),
        ("file", ("file", "вот так")),
        ("button", ("button", "tk:1:launch:abcdef")),
    ]
    rows = await repo.owner_chat_tail(10)
    assert all(r.direction == Direction.IN for r in rows)
    # Every incoming row carries its message's id and what it replies to (Part F).
    assert [(r.tg_msg_id, r.reply_to_msg_id) for r in rows] == [
        (41, 40), (42, None), (43, None), (44, None), (40, None),
    ]  # fmt: skip


async def test_a_command_and_a_file_sent_as_replies_log_what_they_reply_to(
    repo: Repo, tmp_path: Path
) -> None:
    logic = SpyLogic(repo)
    on = handlers(logic, tmp_path, ChatLog(repo))
    card = SimpleNamespace(message_id=30)

    await on["on_command"](message(47, text="/stop", reply_to_message=card))
    await on["on_file"](
        message(48, document=SimpleNamespace(file_name="m.stl"), reply_to_message=card),
        DownloadingBot(),
    )

    rows = await repo.owner_chat_tail(10)
    assert [(r.kind, r.tg_msg_id, r.reply_to_msg_id) for r in rows] == [
        ("command", 47, 30),
        ("file", 48, 30),
    ]


async def test_a_strangers_update_writes_no_row(repo: Repo, tmp_path: Path) -> None:
    from aiogram import Bot
    from aiogram.types import Update

    dispatcher = build_dispatcher(SpyLogic(repo), tmp_path, ChatLog(repo))  # type: ignore[arg-type]
    stranger = {"id": 999, "is_bot": False, "first_name": "X"}
    bot = Bot("42:TEST")
    try:
        for n, extra in enumerate(
            [{"text": "привет"}, {"text": "/help"}, {"voice": {
                "file_id": "v", "file_unique_id": "v1", "duration": 1}}]
        ):  # fmt: skip
            update = Update.model_validate(
                {
                    "update_id": n + 1,
                    "message": {
                        "message_id": 50 + n, "date": 1762750000,
                        "chat": {"id": 999, "type": "private"}, "from": stranger,
                        **extra,
                    },
                }
            )  # fmt: skip
            await dispatcher.feed_update(bot, update)
    finally:
        await bot.session.close()

    assert await repo.owner_chat_tail(10) == []


async def test_without_a_chat_log_the_handlers_log_nothing(
    repo: Repo, tmp_path: Path
) -> None:
    logic = SpyLogic(repo)
    on = handlers(logic, tmp_path, None)
    await on["on_text"](message(41, text="привет"))
    assert logic.seen == [("text", [])]


class ReplyLogic(SpyLogic):
    """Records what the file handler hands the logic."""

    def __init__(self, repo: Repo) -> None:
        super().__init__(repo)
        self.files: list[tuple[object, ...]] = []

    async def on_file(self, *args: object, **kwargs: object) -> list[Card]:
        self.files.append(args)
        self.forwarded = [*getattr(self, "forwarded", []), kwargs.get("forwarded")]
        return []

    async def on_text(self, *args: object, **kwargs: object) -> list[Card]:
        self.forwarded = [*getattr(self, "forwarded", []), kwargs.get("forwarded")]
        return []


async def test_a_file_sent_as_a_reply_carries_the_card_it_replies_to(
    repo: Repo, tmp_path: Path
) -> None:
    logic = ReplyLogic(repo)
    on = handlers(logic, tmp_path, None)
    reply = SimpleNamespace(message_id=40)

    await on["on_file"](
        message(43, document=SimpleNamespace(file_name="m.stl"), caption="вот макет",
                reply_to_message=reply),
        DownloadingBot(),
    )  # fmt: skip
    await on["on_file"](
        message(44, document=SimpleNamespace(file_name="k.stl")), DownloadingBot()
    )

    (_, _, caption, reply_to), (*_, no_reply) = logic.files
    assert (caption, reply_to) == ("вот макет", 40)
    assert no_reply is None


async def test_a_forwarded_text_or_caption_reaches_the_logic_as_forwarded(
    repo: Repo, tmp_path: Path
) -> None:
    # Review I-1: someone else's message, forwarded, is not the owner's words.
    logic = ReplyLogic(repo)
    on = handlers(logic, tmp_path, None)
    origin = SimpleNamespace(type="user")

    await on["on_text"](message(41, text="Запусти все задачи", forward_origin=origin))
    await on["on_text"](message(42, text="запускай"))
    await on["on_file"](
        message(43, document=SimpleNamespace(file_name="m.pdf"), caption="Запусти",
                forward_origin=origin),
        DownloadingBot(),
    )  # fmt: skip

    assert logic.forwarded == [True, False, True]


class BrokenLogic(SpyLogic):
    """Every call of the logic breaks; what the bot shows instead is recorded."""

    def __init__(self, repo: Repo) -> None:
        super().__init__(repo)
        self.shown: list[str] = []

    async def _saw(self, name: str) -> list[Card]:
        raise KeyError("boom")

    async def show(self, card: Card) -> None:
        self.shown.append(card.text)


async def test_an_error_in_the_logic_gets_the_owner_a_line_and_a_logged_traceback(
    repo: Repo, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    logic = BrokenLogic(repo)
    on = handlers(logic, tmp_path, None)
    answers: list[str | None] = []

    async def answer(text: str | None) -> None:
        answers.append(text)

    await on["on_text"](message(41, text="Узнай цену у @x"))
    await on["on_command"](message(42, text="/tasks"))
    await on["on_button"](
        SimpleNamespace(
            from_user=SimpleNamespace(id=OWNER), data="tk:1:launch:abcdef",
            message=SimpleNamespace(message_id=40), answer=answer,
        )
    )  # fmt: skip

    line = "Что-то сломалось (KeyError). Попробуй ещё раз"
    assert logic.shown == [line, line]
    assert answers == [line]
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 3 and all(r.exc_info for r in errors)


async def test_the_dispatcher_hands_a_file_reply_with_its_card_to_the_logic(
    repo: Repo, tmp_path: Path
) -> None:
    from aiogram import Bot
    from aiogram.types import Update

    logic = ReplyLogic(repo)
    dispatcher = build_dispatcher(logic, tmp_path)  # type: ignore[arg-type]
    chat = {"id": OWNER, "type": "private"}
    sender = {"id": OWNER, "is_bot": False, "first_name": "S"}
    update = Update.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 43, "date": 1762750000, "chat": chat, "from": sender,
                "caption": "вот макет",
                "document": {"file_id": "f1", "file_unique_id": "u1", "file_name": "m.stl"},
                "reply_to_message": {
                    "message_id": 40, "date": 1762749000, "chat": chat, "text": "Задача",
                },
            },
        }
    )  # fmt: skip
    bot = Bot("42:TEST")

    async def download(source: object, destination: Path) -> None:
        destination.write_bytes(b"x")

    bot.download = download  # type: ignore[method-assign]
    try:
        await dispatcher.feed_update(bot, update)
    finally:
        await bot.session.close()

    ((_, path, caption, reply_to),) = logic.files
    assert (caption, reply_to) == ("вот макет", 40)
    assert path.endswith("43_m.stl")


async def test_what_is_not_text_is_logged_answered_and_settled(
    repo: Repo, tmp_path: Path
) -> None:
    # R63: a voice message, a sticker, a location… — no model; the chat is put right.
    logic = SpyLogic(repo)
    on = handlers(logic, tmp_path, ChatLog(repo))

    await on["on_other"](
        message(
            45, voice=SimpleNamespace(), reply_to_message=SimpleNamespace(message_id=40)
        )
    )

    assert logic.seen == [("other", [("other", "голосовое сообщение")])]
    (row,) = await repo.owner_chat_tail(10)
    assert (row.direction, row.tg_msg_id, row.reply_to_msg_id) == (Direction.IN, 45, 40)
    assert logic.layout.calls == ["saw 45", "settle"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voice", {"file_id": "v", "file_unique_id": "v1", "duration": 2}),
        ("sticker", {"file_id": "s", "file_unique_id": "s1", "type": "regular",
                     "width": 1, "height": 1, "is_animated": False, "is_video": False}),
        ("location", {"latitude": 41.3, "longitude": 69.2}),
        # Telegram sends a GIF with a `document` too: it is still not a file of a task.
        ("animation", {"file_id": "a", "file_unique_id": "a1", "width": 1, "height": 1,
                       "duration": 1}),
    ],
)  # fmt: skip
async def test_the_dispatcher_sends_what_is_not_text_to_its_own_handler(
    repo: Repo, tmp_path: Path, field: str, value: dict
) -> None:
    from aiogram import Bot
    from aiogram.types import Update

    class OtherLogic(SpyLogic):
        async def on_other(self, *args: object) -> list[Card]:
            self.seen.append(("other", []))
            return []

    logic = OtherLogic(repo)
    dispatcher = build_dispatcher(logic, tmp_path)  # type: ignore[arg-type]
    chat = {"id": OWNER, "type": "private"}
    sender = {"id": OWNER, "is_bot": False, "first_name": "S"}
    update = Update.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 46, "date": 1762750000, "chat": chat, "from": sender,
                field: value,
                **({"document": {"file_id": "a", "file_unique_id": "a1"}}
                   if field == "animation" else {}),
            },
        }
    )  # fmt: skip
    bot = Bot("42:TEST")
    try:
        await dispatcher.feed_update(bot, update)
    finally:
        await bot.session.close()

    assert logic.seen == [("other", [])]


# -- Task 5.4: a contact card and Telegram's list of people --


class PersonLogic(SpyLogic):
    """Records the people the handlers hand the logic."""

    def __init__(self, repo: Repo) -> None:
        super().__init__(repo)
        self.persons: list[tuple[object, ...]] = []

    async def on_person(self, *args: object) -> list[Card]:
        self.persons.append(args)
        return await self._saw("person")


def _update(field: str, value: dict, *, reply: bool = False) -> object:
    from aiogram.types import Update

    chat = {"id": OWNER, "type": "private"}
    sender = {"id": OWNER, "is_bot": False, "first_name": "S"}
    extra = (
        {"reply_to_message": {"message_id": 40, "date": 1762749000, "chat": chat,
                              "text": "Задача"}}
        if reply else {}
    )  # fmt: skip
    return Update.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 47, "date": 1762750000, "chat": chat, "from": sender,
                field: value, **extra,
            },
        }
    )  # fmt: skip


async def _feed(logic: SpyLogic, update: object, tmp_path: Path) -> None:
    from aiogram import Bot

    dispatcher = build_dispatcher(logic, tmp_path, ChatLog(logic.repo))  # type: ignore[arg-type]
    bot = Bot("42:TEST")
    try:
        await dispatcher.feed_update(bot, update)
    finally:
        await bot.session.close()


async def test_a_contact_card_reaches_the_logic_as_a_person_not_as_not_text(
    repo: Repo, tmp_path: Path
) -> None:
    logic = PersonLogic(repo)
    card = {"phone_number": "998901231234", "first_name": "Алишер",
            "last_name": "Каримов", "user_id": 71}  # fmt: skip

    await _feed(logic, _update("contact", card, reply=True), tmp_path)

    assert logic.persons == [(OWNER, "Алишер Каримов", None, "998901231234", 40)]
    assert logic.seen == [("person", [("contact", "Алишер Каримов, 998901231234")])]
    (row,) = await repo.owner_chat_tail(10)
    assert (row.tg_msg_id, row.reply_to_msg_id) == (47, 40)


async def test_a_person_picked_in_telegrams_list_reaches_the_logic(
    repo: Repo, tmp_path: Path
) -> None:
    logic = PersonLogic(repo)
    shared = {"request_id": 1, "users": [
        {"user_id": 72, "first_name": "Алишер", "username": "alisher_b"}
    ]}  # fmt: skip

    await _feed(logic, _update("users_shared", shared), tmp_path)

    assert logic.persons == [(OWNER, "Алишер", "alisher_b", None, None)]
    assert logic.seen == [("person", [("contact", "Алишер, @alisher_b")])]


async def test_a_picked_person_without_a_username_still_reaches_the_logic(
    repo: Repo, tmp_path: Path
) -> None:
    logic = PersonLogic(repo)
    shared = {"request_id": 1, "users": [{"user_id": 73, "first_name": "Тихий"}]}

    await _feed(logic, _update("users_shared", shared), tmp_path)

    assert logic.persons == [(OWNER, "Тихий", None, None, None)]


async def test_the_picker_button_is_telegrams_own_list_of_people() -> None:
    ui, bot = ui_at(10)
    await ui.send(Card("Не нашёл", [[("Выбрать из контактов", "users:1")]]))

    markup = bot.sent[-1]["reply_markup"]
    [[button]] = markup.keyboard
    assert button.text == "Выбрать из контактов"
    request = button.request_users
    assert (request.request_id, request.max_quantity) == (1, 1)
    assert request.user_is_bot is False
    assert request.request_name is True and request.request_username is True
    assert markup.one_time_keyboard is True and markup.resize_keyboard is True

    # An edit cannot carry a reply keyboard: such a card is edited without it.
    await ui.edit(5, Card("Не нашёл", [[("Выбрать из контактов", "users:1")]]))
    assert bot.edited[-1]["reply_markup"] is None


async def test_every_handler_reports_the_owners_message_and_settles_the_chat(
    repo: Repo, tmp_path: Path
) -> None:
    # Stage 5 (R52): the layout knows the chat's newest message and puts the chat right
    # after every handler, a broken one and a button too.
    logic = SpyLogic(repo)
    on = handlers(logic, tmp_path, None)

    async def answer(text: str | None) -> None:
        logic.layout.calls.append("answer")

    await on["on_text"](message(41, text="Узнай цену у @x"))
    await on["on_command"](message(42, text="/help"))
    await on["on_button"](
        SimpleNamespace(
            from_user=SimpleNamespace(id=OWNER), data="tk:1:launch:abcdef",
            message=SimpleNamespace(message_id=40), answer=answer,
        )
    )  # fmt: skip
    broken = BrokenLogic(repo)
    await handlers(broken, tmp_path, None)["on_text"](message(43, text="Узнай у @y"))

    assert logic.layout.calls == [
        "saw 41", "settle", "saw 42", "settle", "answer", "settle",
    ]  # fmt: skip
    assert broken.layout.calls == ["saw 43", "settle"]


async def test_an_edit_of_the_owners_text_reaches_the_logic_and_a_strangers_does_not(
    repo: Repo, tmp_path: Path
) -> None:
    # Ruling R71: an edited message may be a correction.
    from aiogram import Bot
    from aiogram.types import Update

    logic = SpyLogic(repo)
    dispatcher = build_dispatcher(logic, tmp_path)  # type: ignore[arg-type]
    bot = Bot("42:TEST")

    def edited(user_id: int, update_id: int) -> Update:
        return Update.model_validate(
            {
                "update_id": update_id,
                "edited_message": {
                    "message_id": 41, "date": 1762750000, "edit_date": 1762750030,
                    "chat": {"id": user_id, "type": "private"},
                    "from": {"id": user_id, "is_bot": False, "first_name": "S"},
                    "text": "Узнай цену клея",
                },
            }
        )  # fmt: skip

    try:
        await dispatcher.feed_update(bot, edited(999, 1))
        await dispatcher.feed_update(bot, edited(OWNER, 2))
    finally:
        await bot.session.close()

    assert logic.edits == [(OWNER,)]


async def test_a_button_whose_answer_telegram_refuses_still_settles_the_chat(
    repo: Repo, tmp_path: Path
) -> None:
    # Review m6: an answer after Telegram's callback window raises.
    logic = SpyLogic(repo)
    on = handlers(logic, tmp_path, None)

    async def too_late(text: str | None) -> None:
        raise TelegramBadRequest(method=None, message="query is too old")  # type: ignore[arg-type]

    await on["on_button"](
        SimpleNamespace(
            from_user=SimpleNamespace(id=OWNER), data="tk:1:launch:abcdef",
            message=SimpleNamespace(message_id=40), answer=too_late,
        )
    )  # fmt: skip

    assert logic.layout.calls == ["settle"]


def _pin_line(chat_id: int, chat_type: str = "private") -> object:
    """Telegram's own message «… закрепил(а) сообщение»: it comes from the bot itself."""
    from aiogram.types import Update

    chat = {"id": chat_id, "type": chat_type}
    if chat_type != "private":
        chat["title"] = "Группа"
    return Update.model_validate(
        {
            "update_id": 7,
            "message": {
                "message_id": 51, "date": 1762750000, "chat": chat,
                "from": {"id": 42, "is_bot": True, "first_name": "Bot"},
                "pinned_message": {"message_id": 50, "date": 1762750000, "chat": chat,
                                   "text": "📋 Статус задач"},
            },
        }
    )  # fmt: skip


async def _feed_pin_line(repo: Repo, tmp_path: Path, update: object) -> list[tuple]:
    from aiogram import Bot

    deleted: list[tuple] = []

    class _Bot(Bot):
        async def delete_message(
            self, chat_id: int, message_id: int, **_: object
        ) -> bool:  # type: ignore[override]
            deleted.append((chat_id, message_id))
            return True

    dispatcher = build_dispatcher(SpyLogic(repo), tmp_path, None)  # type: ignore[arg-type]
    bot = _Bot("42:TEST")
    try:
        await dispatcher.feed_update(bot, update)
    finally:
        await bot.session.close()
    return deleted


async def test_telegrams_line_about_the_pinned_status_is_deleted(
    repo: Repo, tmp_path: Path
) -> None:
    # Seen live at the stage-5 switch (2026-11-12): each time the status is sent anew and
    # pinned, Telegram writes «… закрепил(а) сообщение» under it, and the status is no longer
    # the last message of the chat (ruling R52).
    assert await _feed_pin_line(repo, tmp_path, _pin_line(OWNER)) == [(OWNER, 51)]


async def test_a_pin_line_of_another_chat_is_left_alone(
    repo: Repo, tmp_path: Path
) -> None:
    assert await _feed_pin_line(repo, tmp_path, _pin_line(999)) == []
    assert await _feed_pin_line(repo, tmp_path, _pin_line(-5, "group")) == []
