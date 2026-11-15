"""The whole program on fakes: from the owner's text to the result.

`main.assemble` wires the program as it runs (`inline_wakes=True`: each wake runs inside the
Scheduler's round), on a real Repo, a fake Telegram, a fake bot and a scripted model. Each test
is one story, told the way it happens: the owner writes to the bot and presses its buttons
(through `ControlLogic`, the way the bot's handlers call it), people write in Telegram (through
the userbot's callbacks, `app.events`), and time passes in Scheduler rounds. What is asserted is
what the fake bot showed and what the fake Telegram sent.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

import aiohttp
import pytest
from agent_world import BURST, DELIVERY, END, ask, call, send

from tg_agent.brain.llm import FakeLLM, LLMError
from tg_agent.clock import FakeClock
from tg_agent.config import Remote, Settings
from tg_agent.control.base import Card, FakeUI
from tg_agent.db import Repo
from tg_agent.gateway.base import Incoming, PeerInfo
from tg_agent.gateway.fake import FakeGateway
from tg_agent.main import App, assemble, start_up
from tg_agent.models import (
    ApprovalStatus,
    ConvState,
    EventKind,
    OutboxStatus,
    TaskStatus,
)
from tg_agent.search.agent import Candidate, SearchOutcome, Wall
from tg_agent.search.fake import FakeSearch
from tg_agent.search.remote import RemoteKey, seal
from tg_agent.search.remote_server import RemoteServer

OWNER = 111
PEER = 1001
TITLE = "3D-печать"
ASKED = "Сколько стоит печать 5 корпусов из PETG?"
GREETING = f"Здравствуйте! {ASKED}"


class Telegram(FakeGateway):
    """The fake userbot: it can be started like the real one, and its catch-up (`replay`)
    reports what reached a chat after a message, as the real one does: a person's message
    as incoming, an outgoing one the program did not store as the owner's own."""

    events: Any = None  # the program's inbox, set when the program is built

    async def start(self) -> None:
        self.calls.append("start")

    async def replay(
        self, peer_id: int, after_msg_id: int, known_out_ids: set[int]
    ) -> int:
        await super().replay(peer_id, after_msg_id, known_out_ids)
        count = 0
        for message in self.chats.get(peer_id, []):
            if message.tg_msg_id <= after_msg_id:
                continue
            if not message.out:
                await self.events.on_incoming(
                    Incoming(
                        peer_id, message.tg_msg_id, message.text, "text", message.date
                    )
                )
            elif message.tg_msg_id not in known_out_ids:
                await self.events.on_owner_manual(
                    peer_id, message.text, message.tg_msg_id
                )
            else:
                continue
            count += 1
        return count


def draft(
    *targets: str,
    task_id: int | None = None,
    title: str = TITLE,
    brief: str = "Узнать цену печати 5 корпусов",
    uses_files: bool = False,
) -> dict[str, Any]:
    """The secretary's model drafts a task (or rewrites draft `task_id`)."""
    return call(
        "draft_task", task_id=task_id, title=title, brief=brief, people=list(targets),
        ceiling=300_000, currency="UZS", share=["name"], missing=[],
        uses_files=uses_files,
    )  # fmt: skip


def say(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text}]


def notes(person: int, headline: str, note: str) -> dict[str, Any]:
    return call(
        "update_notes",
        notes=headline,
        headline=headline,
        person_notes=[{"person_id": person, "note": note}],
    )


class Program:
    """The program built by `main.assemble`, and the moves of a story."""

    def __init__(
        self,
        repo: Repo,
        clock: FakeClock,
        settings: Settings,
        llm: FakeLLM,
        **assembled: Any,
    ) -> None:
        self.repo, self.clock, self.settings, self.llm = repo, clock, settings, llm
        self.assembled = assembled  # more of `assemble`'s arguments (a scripted search)
        self.telegram = Telegram(clock)
        self.telegram.peers["print_lab"] = PeerInfo(PEER, "print_lab", "Акмаль", False)
        self.bot = FakeUI()
        self.app = self.build()
        self.task_id = 0
        self.person = 0  # the conversation id: the agent's «person_id»

    def build(self) -> App:
        """The program, started from scratch on the same database (a restart)."""
        app = assemble(
            self.settings,
            self.clock,
            self.repo,
            self.llm,
            self.telegram,
            self.bot,
            canary="ZX-flow-0001",
            inline_wakes=True,
            **self.assembled,
        )
        self.telegram.events = app.events
        return app

    # -- the owner --

    async def show(self, cards: list[Card]) -> list[int]:
        """What the bot does with the cards the logic returns."""
        return [await self.app.logic.show(card) for card in cards]

    # As the bot's handlers do (stage 5): the owner's message is the chat's newest, and the
    # chat's layout is put right once the logic is done (`ChatLayout.settle`).

    async def write(self, text: str, reply_to: int | None = None) -> list[Card]:
        await self.app.logic.layout.saw(self.bot.owner_says(text))
        cards = await self.app.logic.on_text(OWNER, text, reply_to, [])
        await self.show(cards)
        await self.app.logic.layout.settle()
        return cards

    async def command(self, name: str, args: str = "") -> list[Card]:
        await self.app.logic.layout.saw(self.bot.owner_says(f"/{name} {args}".strip()))
        cards = await self.app.logic.on_command(OWNER, name, args)
        await self.show(cards)
        await self.app.logic.layout.settle()
        return cards

    async def send_file(
        self, path: str, caption: str = "", reply_to: int | None = None
    ) -> list[Card]:
        await self.app.logic.layout.saw(self.bot.owner_says(caption or path))
        cards = await self.app.logic.on_file(OWNER, path, caption, reply_to)
        await self.show(cards)
        await self.app.logic.layout.settle()
        return cards

    async def other(self, what: str) -> list[Card]:
        """A voice message, a sticker, a location…: what the bot's handler for them does."""
        await self.app.logic.layout.saw(self.bot.owner_says(what))
        cards = await self.app.logic.on_other(OWNER)
        await self.show(cards)
        await self.app.logic.layout.settle()
        return cards

    async def press(self, data: str) -> str:
        toast = await self.app.logic.on_callback(OWNER, data)
        await self.app.logic.layout.settle()
        return toast

    def msg_id(self, card: Card) -> int:
        """The message id the bot gave this card."""
        return self.bot.sent_ids[self.bot.sent.index(card)]

    def cards(self) -> list[Card]:
        """Every message the bot sent the owner, but the pinned board."""
        return [card for card in self.bot.sent if card.kind != "board"]

    def board(self) -> str:
        """The pinned board as it reads now."""
        shown = [
            (msg_id, card)
            for msg_id, card in zip(self.bot.sent_ids, self.bot.sent, strict=True)
            if card.kind == "board"
        ]
        assert shown, "no board was sent"
        msg_id, card = shown[-1]
        assert self.bot.pins[-1] == msg_id
        edits = [card for edited, card in self.bot.edits if edited == msg_id]
        return (edits[-1] if edits else card).text

    # -- the person --

    async def person_writes(self, text: str) -> None:
        message = self.telegram.add_incoming(PEER, text)
        await self.app.events.on_incoming(
            Incoming(PEER, message.tg_msg_id, text, "text", message.date)
        )

    def sent(self) -> list[str]:
        return [text for peer, text in self.telegram.sent if peer == PEER]

    # -- time --

    async def round(self, advance: timedelta | None = None) -> None:
        """Time passes, then one round of the Scheduler."""
        if advance is not None:
            self.clock.advance(advance)
        await self.app.scheduler.tick()

    async def until(
        self, done: Callable[[], Awaitable[bool] | bool], step: timedelta
    ) -> None:
        for _ in range(200):
            outcome = done()
            if outcome if isinstance(outcome, bool) else await outcome:
                return
            await self.round(step)
        raise AssertionError("it never happened")

    # -- the model --

    def script(self, *steps: list[dict[str, Any]]) -> None:
        self.llm.steps["task"].extend(steps)

    def secretary(self, *steps: object) -> None:
        """Script the secretary's model for the owner's next message."""
        self.llm.steps["secretary"].extend(steps)

    def asked(self) -> list[dict[str, Any]]:
        """The secretary's requests to the model."""
        return [req for purpose, req in self.llm.requests if purpose == "secretary"]

    def drafts(self) -> list[tuple[int, Card]]:
        """The draft cards on screen now, oldest first, with their message ids."""
        return [
            (msg_id, card)
            for msg_id, card in self.bot.chat()
            if card.kind == "task" and card.text.startswith("❓ Новая задача")
        ]

    async def drafted(self, text: str, *steps: object) -> Card:
        """The owner writes a task; the secretary drafts it: the newest draft card."""
        self.secretary(*steps, [])
        assert await self.write(text) == []
        return self.drafts()[-1][1]

    def steps(self) -> list[dict[str, Any]]:
        """The model's requests of the task agent: one per step of each wake."""
        return [request for purpose, request in self.llm.requests if purpose == "task"]

    def context(self) -> str:
        """The data the newest wake was given (its first user message)."""
        return self.steps()[-1]["messages"][0]["content"][-1]["text"]

    def trusted(self) -> str:
        """The trusted part of the newest wake's request: the task, the owner's words."""
        return self.steps()[-1]["system"][1]

    async def wakes(self) -> int:
        """How many wakes have started."""
        rows = await self.repo._all("SELECT id FROM wakes")
        return len(rows)

    async def assert_clean(self) -> None:
        assert not self.llm.steps["task"], "scripted steps left unused"
        failed = await self.repo.events_of("wake_failed")
        assert failed == [], failed

    # -- the openings --

    async def launched(self) -> Card:
        """The owner writes the task and presses «Запустить»; returns the «запущена» card:
        the draft's card, edited in place (stage 5, R55)."""
        card = await self.drafted(
            "Узнай цену печати 5 корпусов из PETG у @print_lab", [draft("print_lab")]
        )
        self.task_id = card.task_id
        launch = next(
            data for row in card.buttons for _, data in row if ":launch:" in data
        )
        assert await self.press(launch) == "Запущено"
        (conv,) = await self.repo.task_conversations(self.task_id)
        self.person = conv.id
        return self.bot.screen[self.msg_id(card)]

    async def contacted(self) -> None:
        """Launched, and the first message is in the chat."""
        await self.launched()
        self.script([send(self.person, GREETING)], END)
        await self.round()
        await self.until(lambda: bool(self.sent()), timedelta(minutes=1))

    async def talking(self, text: str = "Есть, 240 000 сум, 3 дня. Берёте?") -> None:
        """Contacted, and the person answered; the wake for his answer is not run yet."""
        await self.contacted()
        await self.round(timedelta(minutes=10))
        await self.person_writes(text)


@pytest.fixture
def program(repo: Repo, fake_clock: FakeClock, settings: Settings) -> Program:
    return Program(repo, fake_clock, settings, FakeLLM())


async def test_the_owners_text_becomes_a_launched_task_and_a_first_message(
    program: Program,
) -> None:
    launched = await program.launched()

    assert launched.text == f"Задача #{program.task_id} «{TITLE}» запущена"
    assert [label for row in launched.buttons for label, _ in row] == ["Закрыть задачу"]
    assert program.sent() == []  # nothing goes before the agent's first wake

    program.script([send(program.person, GREETING)], END)
    await program.round()
    await program.until(lambda: bool(program.sent()), timedelta(minutes=1))

    (first,) = program.sent()
    # The agent's words, with the AI introduction the code puts after the greeting.
    assert first.startswith("Здравствуйте!") and first.endswith(ASKED)
    assert "ИИ" in first
    board = program.board()
    assert f"#{program.task_id} {TITLE} — в работе" in board
    assert "• @print_lab — жду ответа с" in board
    # The owner got the task card, which became the «запущена» card (stage 5, R55),
    # and nothing else but the status.
    assert [card.kind for card in program.cards()] == ["task"]
    assert program.bot.notes == []
    await program.assert_clean()


async def test_a_task_without_contacts_runs_with_nobody_in_it_and_closes(
    program: Program,
) -> None:
    # Ruling R26: the owner names nobody, presses «Запустить», and the agent's first wake
    # runs with no people in the task (the search tools come with Task 4.3).
    card = await program.drafted("Найди, кто напечатает 5 корпусов из PETG", [draft()])
    task_id = card.task_id
    assert "4. Контакты\nнайду сам" in card.text
    launch = next(data for row in card.buttons for _, data in row if ":launch:" in data)
    assert await program.press(launch) == "Запущено"
    assert program.bot.screen[program.msg_id(card)].text == (
        f"Задача #{task_id} «{TITLE}» запущена\n"
        "Кому писать — ищу сам. Ход виден в закреплённом статусе."
    )
    headline = "Ищу мастерские 3D-печати PETG в Ташкенте"
    program.script(
        [call("update_notes", notes=headline, headline=headline, person_notes=[])],
        END,
    )

    await program.round()

    assert await program.wakes() == 1
    assert "Ищу мастерские" not in program.context()  # the wake saw an empty task
    board = program.board()
    assert board.splitlines()[-3:] == [
        f"#{task_id} {TITLE} — в работе",
        headline,
        "• людей пока нет",  # nobody in the task
    ]
    shown = next(c for c in reversed(program.bot.sent) if c.kind == "board")
    assert shown.buttons == [[(f"Закрыть #{task_id} {TITLE}", f"fn:{task_id}:ask:-")]]
    await program.assert_clean()

    # The keep-alive handles a task with nobody in it as any other: nothing is scheduled,
    # so the agent gets its nudge (a reminder in ten minutes).
    assert await program.repo.events_of("task_nudged") == [{"task_id": task_id}]

    assert await program.press(f"fn:{task_id}:ask:-") == ""
    confirm = program.bot.sent[-1]
    yes = next(data for row in confirm.buttons for _, data in row if ":yes:" in data)
    assert await program.press(yes) == f"Задача #{task_id} закрыта"
    assert program.board() == "📋 Задач в работе нет"
    assert (await program.repo.get_task(task_id)).status is TaskStatus.CANCELLED
    assert program.telegram.sent == []
    await program.round(timedelta(minutes=11))  # the nudge is not for a closed task
    assert await program.wakes() == 1
    await program.assert_clean()


async def test_the_person_answers_and_the_owner_decides_by_replying_to_the_card(
    program: Program,
) -> None:
    await program.talking()
    before = len(program.cards())
    program.script(
        [
            notes(program.person, "PrintLab: 240 000 сум, 3 дня", "240 000, 3 дня"),
            ask(
                "question",
                "PrintLab: 240 000 сум, 3 дня. Брать?",
                program.person,
                suggested=["Беру, когда можно забрать?"],
            ),
        ],
        END,
    )

    await program.round(BURST)

    board = program.board()
    assert "PrintLab: 240 000 сум, 3 дня" in board.splitlines()
    assert "• @print_lab — ждёт твоего решения · 240 000, 3 дня" in board
    (question,) = program.cards()[before:]
    assert question.kind == "approval"
    assert "PrintLab: 240 000 сум, 3 дня. Брать?" in question.text
    assert program.bot.notes == []

    program.script([send(program.person, "Подскажите, доставка завтра возможна?")], END)
    (answer,) = await program.write(
        "Сначала спроси про доставку завтра", program.msg_id(question)
    )
    assert answer.text == "Ответ передан агенту"
    await program.round()
    await program.round(DELIVERY)

    assert program.sent()[-1] == "Подскажите, доставка завтра возможна?"
    assert "Сначала спроси про доставку завтра" in program.trusted()
    assert (await program.repo.get_conversation(program.person)).state is (
        ConvState.TALKING
    )
    await program.assert_clean()


async def test_the_result_comes_once_and_a_reply_to_it_takes_the_task_up_again(
    program: Program,
) -> None:
    await program.talking()
    before = len(program.cards())
    program.script(
        [call("finish", result="PrintLab: 240 000 сум, 3 дня. Можно заказывать.")]
    )

    await program.round(BURST)

    (result,) = program.cards()[before:]
    assert result.text == (
        f"✅ Готово — {TITLE}\nPrintLab: 240 000 сум, 3 дня. Можно заказывать."
    )
    # R109: in work until the owner ends it; the board shows it ready
    assert (await program.repo.get_task(program.task_id)).status is TaskStatus.ACTIVE
    assert f"#{program.task_id} {TITLE} — готова" in program.board()

    (taken_up,) = await program.write("уточни срок", program.msg_id(result))
    assert taken_up.text == f"Задача #{program.task_id} снова в работе: передал агенту"
    program.script(END)
    await program.round()

    assert f"#{program.task_id} {TITLE} — в работе" in program.board()
    trusted = program.trusted()
    instructions = trusted[trusted.index("# Указания") :]
    assert "уточни срок" in instructions
    await program.assert_clean()


async def test_closing_from_the_board_says_goodbye_and_stops_listening(
    program: Program,
) -> None:
    await program.talking()
    program.script(END)
    await program.round(BURST)
    asked = await program.wakes()
    board = next(card for card in reversed(program.bot.sent) if card.kind == "board")
    [close] = [data for row in board.buttons for _, data in row]

    assert await program.press(close) == ""
    confirm = program.bot.sent[-1]
    assert confirm.text.startswith(f"❓ Закрыть задачу #{program.task_id} «{TITLE}»?")
    yes = next(data for row in confirm.buttons for _, data in row if ":yes:" in data)
    toast = await program.press(yes)

    assert toast == f"Задача #{program.task_id} закрыта. Попрощался: @print_lab"
    await program.round(DELIVERY)
    assert program.sent()[-1] == "Спасибо, пока не нужно"
    assert program.board() == "📋 Задач в работе нет"

    await program.person_writes("А почему?")
    await program.round(BURST)
    await program.round(timedelta(hours=1))

    assert await program.wakes() == asked
    assert program.sent()[-1] == "Спасибо, пока не нужно"
    assert (await program.repo.get_task(program.task_id)).status is (
        TaskStatus.CANCELLED
    )
    await program.assert_clean()


async def test_a_paused_task_answers_only_after_the_resume(program: Program) -> None:
    await program.contacted()
    asked = await program.wakes()

    task = program.task_id
    program.secretary(
        [call("set_task_state", task_id=task, state="pause", quote="на паузу")], []
    )
    assert await program.write(f"поставь задачу {task} на паузу") == []
    paused = program.asked()[-1]["messages"][-1]["content"][0]["content"]
    assert paused == (
        f"Пауза в задаче #{task}. Продолжить — напиши «продолжи задачу {task}»"
    )
    await program.round(timedelta(minutes=10))
    await program.person_writes("Есть, 240 000 сум")
    await program.round(BURST)
    await program.round(DELIVERY)

    assert await program.wakes() == asked
    assert program.sent() == [program.sent()[0]]  # the greeting only
    assert "на паузе" in program.board()

    program.script([send(program.person, "Спасибо! А сроки какие?")], END)
    program.secretary(
        [call("set_task_state", task_id=task, state="resume", quote="продолжи")],
        say(f"Задача #{task} продолжена."),
    )
    (resumed,) = await program.write(f"продолжи задачу {task}")
    assert resumed.text == f"Задача #{task} продолжена."
    assert (await program.repo.get_task(task)).status is TaskStatus.ACTIVE
    await program.round()
    await program.round(DELIVERY)

    assert await program.wakes() == asked + 1
    assert "Есть, 240 000 сум" in program.context()
    assert program.sent()[-1] == "Спасибо! А сроки какие?"
    await program.assert_clean()


async def test_stop_holds_wakes_and_sends_and_resume_all_lets_them_go(
    program: Program,
) -> None:
    await program.talking()
    program.script([send(program.person, "Спасибо! А сроки какие?")], END)
    await program.round(BURST)  # the reply is queued

    (stopped,) = await program.command("stop")
    assert stopped.text == "Всё остановлено. Продолжить — кнопка под закреплённым статусом"
    await program.round(DELIVERY)
    assert program.sent()[-1] == program.sent()[0]  # held: the greeting is the last

    assert await program.press("rs:0:go:-") == "Работаю дальше"  # under the status
    await program.round()
    assert program.sent()[-1] == "Спасибо! А сроки какие?"

    await program.command("stop")
    await program.round(timedelta(minutes=10))
    await program.person_writes("3 дня")
    asked = await program.wakes()
    await program.round(BURST)
    await program.round(timedelta(minutes=30))
    assert await program.wakes() == asked  # no wake while stopped

    program.script(END)
    program.secretary([call("set_running", on=True, quote="продолжай")], [])
    await program.write("продолжай")  # or in words
    await program.round()
    assert await program.wakes() == asked + 1
    assert "3 дня" in program.context()
    await program.assert_clean()


async def test_a_task_launched_while_stopped_says_so_and_goes_on_the_resume_button(
    program: Program,
) -> None:
    (stopped,) = await program.command("stop")
    assert "кнопка под закреплённым статусом" in stopped.text
    assert program.board().split("\n")[1] == (
        "⏸ Агент остановлен — ничего не пишет и не отвечает"
    )

    launched = await program.launched()
    assert launched.text == (
        f"Задача #{program.task_id} «{TITLE}» запущена" + chr(10)
        + "Агент сейчас остановлен: нажми «Продолжить работу» под закреплённым"
        " статусом — и он начнёт."
    )  # fmt: skip
    await program.round(timedelta(minutes=30))
    assert await program.wakes() == 0 and program.sent() == []

    assert await program.press("rs:0:go:-") == "Работаю дальше"
    assert "⏸" not in program.board()
    program.script([send(program.person, GREETING)], END)
    await program.round()
    await program.until(lambda: bool(program.sent()), timedelta(minutes=1))
    assert program.sent()[0].startswith("Здравствуйте!")
    await program.assert_clean()


async def test_an_evening_with_nothing_waiting_sends_nothing(program: Program) -> None:
    program.clock.advance(timedelta(hours=10, minutes=30))  # 20:30
    await program.round()
    await program.round(timedelta(minutes=5))
    assert program.bot.sent == [] and program.bot.notes == []
    assert await program.repo.get_kv("digest_sent_on") == "2026-11-10"


async def test_a_card_waiting_for_hours_comes_back_once_in_the_evening(
    program: Program,
) -> None:
    card = await program.drafted("Найди, кто напечатает 5 корпусов", [draft()])
    program.clock.advance(timedelta(hours=10, minutes=30))  # 20:30: it waited 10 hours
    await program.round()
    await program.round(timedelta(minutes=5))

    [(msg_id, again)] = program.drafts()
    assert again.task_id == card.task_id and program.bot.last() is again
    assert msg_id not in program.bot.silent  # with sound: the reminder
    resent = [c for c in program.bot.sent if c.task_id == card.task_id]
    assert len(resent) == 2  # sent once more, that day only
    assert not [n for n in program.bot.notes if n.startswith("Сводка дня")]


REPLY = "Спасибо! А сроки какие?"


async def crashed(program: Program) -> int:
    """The program stops dead in the middle of its work and leaves, in the database and in
    the chat: (a) a reply whose send was cut short after it reached Telegram (SENDING), (b) a
    wake that claimed its events and never finished, (c) a message the person wrote while the
    program was off. Returns the id of the reply's outbox item."""
    await program.talking()
    program.script([send(program.person, REPLY)], END)
    await program.round(BURST)  # the reply is queued
    (item,) = [
        i for i in await program.repo.outbox_of(program.person) if i.text == REPLY
    ]
    # (a) the Sender took it and Telegram got it; the program died before writing it down.
    assert await program.repo.claim_outbox(item.id)
    await program.telegram.send_text(PEER, REPLY)
    # (b) a reminder's wake had claimed it when the program died.
    await program.repo.add_event(
        program.task_id, EventKind.REMINDER, payload={"about": "проверить, ответил ли"}
    )
    wake_id, _ = await program.repo.claim_events(program.task_id, program.clock.now())
    assert wake_id
    # (c) the person wrote while nothing was listening.
    program.clock.advance(timedelta(minutes=5))
    program.telegram.add_incoming(PEER, "Через неделю. Устроит?")
    return item.id


async def test_a_restart_finishes_what_the_crash_left_and_sends_nothing_twice(
    program: Program,
) -> None:
    item_id = await crashed(program)

    program.app = program.build()  # a new program on the same database
    await start_up(program.app)
    asked = len(program.steps())
    program.script(END, END)
    await program.round()
    await program.round(DELIVERY)

    # (a) found in the chat and written down as sent: not sent again.
    assert (await program.repo.get_outbox(item_id)).status is OutboxStatus.SENT
    assert program.sent().count(REPLY) == 1
    assert sum(text.endswith(ASKED) for text in program.sent()) == 1
    # (b) its events woke the agent, which was told the program restarted.
    assert await program.repo.unfinished_wakes() == []
    contexts = [
        step["messages"][0]["content"][-1]["text"] for step in program.steps()[asked:]
    ]
    assert len(contexts) == 2  # the restarted wake, and the wake for (c)
    restarted, answered = contexts
    assert "проверить, ответил ли" in restarted
    assert "программа перезапускалась посреди работы" in restarted
    # (c) stored, and its own wake once the burst window is over.
    assert "новые сообщения от [1]" in answered
    assert "Собеседник: Через неделю. Устроит?" in answered
    assert (await program.repo.get_conversation(program.person)).paused_until is None
    await program.assert_clean()


# -- stage 5: the chat reads at a glance (rulings R52–R56) --


def screen(program: Program) -> list[Card]:
    """The bot's messages on screen now, oldest first (the owner's own left out)."""
    return [card for _, card in program.bot.chat() if card.kind != "owner"]


def assert_tidy(program: Program) -> None:
    """One status at most, and no card shown twice."""
    shown = screen(program)
    assert sum(card.kind == "board" for card in shown) <= 1
    texts = [card.text for card in shown]
    assert len(texts) == len(set(texts)), texts


async def test_the_owners_story_keeps_the_right_message_last(program: Program) -> None:
    # A task → its draft's card is last.
    card = await program.drafted(
        "Узнай цену печати 5 корпусов из PETG у @print_lab", [draft("print_lab")]
    )
    program.task_id = card.task_id
    assert program.bot.last().text.startswith(f"❓ Новая задача #{card.task_id}")
    assert_tidy(program)

    # «Запустить»: the card turns into «запущена», the status is last.
    launch = next(data for row in card.buttons for _, data in row if ":launch:" in data)
    assert await program.press(launch) == "Запущено"
    draft_id = program.msg_id(card)
    launched = f"Задача #{card.task_id} «{TITLE}» запущена"
    assert program.bot.screen[draft_id].text == launched
    assert program.bot.last().kind == "board"
    assert program.bot.last().text.startswith("📋 Статус задач")
    assert_tidy(program)
    (conv,) = await program.repo.task_conversations(program.task_id)
    program.person = conv.id
    program.script([send(program.person, GREETING)], END)
    await program.round()
    await program.until(lambda: bool(program.sent()), timedelta(minutes=1))
    await program.round(timedelta(minutes=10))
    await program.person_writes("Есть, 240 000 сум, 3 дня. Берёте?")

    # The agent asks: the question is last, the status above it shows the task.
    program.script(
        [ask("question", "PrintLab: 240 000 сум, 3 дня. Брать?", program.person)], END
    )
    await program.round(BURST)
    question_id = program.bot.chat()[-1][0]
    assert program.bot.last().text.startswith("❓ Вопрос")
    status = [card for card in screen(program) if card.kind == "board"]
    assert f"#{program.task_id} {TITLE} — в работе" in status[0].text
    assert "ждёт твоего решения" in status[0].text
    assert_tidy(program)

    # He answers: the card settles, the status is last.
    program.script(END)
    (answer,) = await program.write("Бери", question_id)
    assert answer.text == "Ответ передан агенту"
    assert program.bot.screen[question_id].text.endswith("Ответ: «Бери»")
    assert program.bot.last().kind == "board"
    await program.round()
    assert program.bot.last().kind == "board"
    assert_tidy(program)

    # The result: last, with «Завершить».
    await program.round(timedelta(minutes=5))
    await program.person_writes("Ждём заказ")
    program.script([call("finish", result="PrintLab: 240 000 сум, 3 дня.")])
    await program.round(BURST)
    result = program.bot.last()
    assert result.kind == "result"
    assert result.text.startswith(f"✅ Готово — {TITLE}")
    assert result.buttons == [[("Завершить", f"fn:{program.task_id}:ack:-")]]
    assert_tidy(program)

    # «Завершить»: the status is last and says that no task is in work.
    result_id = program.bot.chat()[-1][0]
    assert await program.press(f"fn:{program.task_id}:ack:-") == (
        f"Задача #{program.task_id} завершена"
    )
    assert program.bot.screen[result_id].buttons == []
    assert program.bot.last().kind == "board"
    assert program.bot.last().text == "📋 Задач в работе нет"
    assert (await program.repo.get_task(program.task_id)).status is TaskStatus.DONE
    assert_tidy(program)
    await program.round()
    assert program.bot.last().text == "📋 Задач в работе нет"
    await program.assert_clean()


# -- stage 5.3: the owner talks to the bot in words (the owner's check list) --


def owner_instructions_of(trusted: str) -> str:
    return trusted[trusted.index("# Указания") :]


async def test_story_what_about_the_tasks_is_a_reply_and_no_draft(
    program: Program,
) -> None:
    await program.launched()
    program.script(END)
    await program.round()
    tasks = len(await program.repo.all_tasks())
    program.secretary([call("overview")], say("В работе одна задача: 3D-печать."))

    (reply,) = await program.write("что по задачам?")

    assert reply.text == "В работе одна задача: 3D-печать."
    assert len(await program.repo.all_tasks()) == tasks  # no draft
    assert program.bot.last().kind == "board"  # the status is the last message
    assert not program.llm.steps["secretary"]
    await program.assert_clean()


async def test_story_thanks_gets_a_short_reply_and_changes_nothing(
    program: Program,
) -> None:
    await program.launched()
    program.script(END)
    await program.round()
    before = [(t.id, t.status) for t in await program.repo.all_tasks()]
    program.secretary(say("Пожалуйста!"))

    (reply,) = await program.write("спасибо")

    assert reply.text == "Пожалуйста!"
    assert [(t.id, t.status) for t in await program.repo.all_tasks()] == before
    assert program.bot.last().kind == "board"
    await program.assert_clean()


async def test_story_two_tasks_in_one_message_make_two_drafts(program: Program) -> None:
    text = "Узнай цену печати у @print_lab. И найди, кто сделает 10 ключей"
    keys = draft(title="Ключи", brief="Найти, кто сделает 10 ключей")

    await program.drafted(text, [draft("print_lab"), keys])

    (_, first), (_, second) = program.drafts()
    assert (first.text.split("\n")[3], second.text.split("\n")[3]) == (TITLE, "Ключи")
    assert program.bot.last() is second  # the newest last
    printing, keys_task = await program.repo.all_tasks()
    assert printing.source_text == "" and keys_task.source_text == ""  # R38
    assert_tidy(program)


async def test_story_a_correction_in_words_edits_the_same_card(
    program: Program,
) -> None:
    card = await program.drafted("Найди, кто напечатает 5 корпусов", [draft()])
    task_id = card.task_id
    program.secretary(
        [draft("print_lab", task_id=task_id, brief="Узнать цену у @print_lab")], []
    )

    await program.write("спроси ещё у @print_lab")  # not a reply to the card

    [(_, fixed)] = program.drafts()  # no second card
    assert fixed.task_id == task_id and fixed.text.startswith(
        f"❓ Новая задача #{task_id}"
    )
    assert "• @print_lab — напишу после запуска" in fixed.text
    assert program.bot.last() is fixed
    assert [t.id for t in await program.repo.all_tasks()] == [task_id]
    assert_tidy(program)


async def test_story_launch_by_words(program: Program) -> None:
    card = await program.drafted(
        "Узнай цену печати 5 корпусов у @print_lab", [draft("print_lab")]
    )
    card_id = program.msg_id(card)
    program.secretary(
        [call("launch_task", task_id=card.task_id, quote="запускай")], []
    )

    assert await program.write("неважно, запускай") == []

    assert program.bot.screen[card_id].text == (
        f"Задача #{card.task_id} «{TITLE}» запущена"
    )
    assert program.bot.last().kind == "board"
    program.script(END)
    await program.round()
    assert await program.wakes() == 1  # the task's agent has its first wake
    await program.assert_clean()


async def test_story_an_instruction_in_words_reaches_the_agent_as_said(
    program: Program,
) -> None:
    await program.contacted()
    quote = "спроси ещё про доставку"
    program.secretary(
        [call("instruct_task", task_id=program.task_id, quote=quote)], say("Передал.")
    )

    (reply,) = await program.write(f"по задаче {program.task_id} {quote}")

    assert reply.text == "Передал."
    program.script(END)
    await program.round()
    instructions = owner_instructions_of(program.trusted())
    assert quote in instructions and f"по задаче {program.task_id}" not in instructions
    await program.assert_clean()


async def test_story_a_reply_to_a_drafts_card_names_the_draft(
    program: Program,
) -> None:
    card = await program.drafted("Найди, кто напечатает 5 корпусов", [draft()])
    program.secretary(say("Понял."))

    await program.write("а сроки?", program.msg_id(card))

    content = program.asked()[-1]["messages"][0]["content"]
    assert f"это ответ на карточку черновика #{card.task_id}" in content


async def test_story_a_text_answers_the_one_open_question(program: Program) -> None:
    await program.talking()
    program.script(
        [ask("question", "PrintLab: 240 000 сум. Брать?", program.person)], END
    )
    await program.round(BURST)
    (approval,) = await program.repo.pending_approvals()
    question_id = program.bot.chat()[-1][0]
    program.secretary(
        [call("answer_question", approval_id=approval.id, quote="бери за 240")], []
    )

    await program.write("да, бери за 240")  # not a reply to the card

    answered = await program.repo.get_approval(approval.id)
    assert (answered.status, answered.answer) == (ApprovalStatus.ANSWERED, "бери за 240")
    assert program.bot.screen[question_id].text.endswith("Ответ: «бери за 240»")
    assert program.bot.last().kind == "board"


async def test_story_words_the_owner_did_not_write_launch_nothing(
    program: Program,
) -> None:
    card = await program.drafted("Найди, кто напечатает 5 корпусов", [draft()])
    program.secretary(
        [call("launch_task", task_id=card.task_id, quote="запускай")],
        say("Не запустил."),
    )

    await program.write("подумаю ещё")

    assert (await program.repo.get_task(card.task_id)).status is TaskStatus.DRAFT
    [(_, still)] = program.drafts()  # the card waits as it was
    assert still.task_id == card.task_id and program.bot.last() is still
    assert "quote" in program.asked()[-1]["messages"][-1]["content"][0]["content"]


async def test_story_claude_is_down_and_a_waiting_file_still_waits(
    program: Program,
) -> None:
    await program.send_file("files/inbox/5_k.stl")
    program.secretary(LLMError("overloaded"))

    (reply,) = await program.write("Напечатай по этому файлу 5 корпусов")

    assert reply.text == "Claude сейчас не отвечает: попробуй через минуту"
    assert await program.repo.all_tasks() == []
    assert await program.repo.get_kv("pending_files") == '["files/inbox/5_k.stl"]'
    assert program.bot.last() is reply


async def test_story_files_reach_the_draft_that_needs_them(program: Program) -> None:
    # A file with a caption: a draft that holds it.
    program.secretary([draft(uses_files=True)], [])
    assert await program.send_file("files/inbox/1_a.stl", "Напечатай 5 таких") == []
    (first,) = await program.repo.all_tasks()
    assert first.attachments == ["files/inbox/1_a.stl"]
    assert "Файлы: a.stl" in program.bot.last().text

    # A file alone, then «спасибо»: the file still waits.
    (saved,) = await program.send_file("files/inbox/2_b.stl")
    assert saved.text == "Файл сохранён. Пришли текст задачи — приложу его к ней."
    program.secretary(say("Пожалуйста!"))
    await program.write("спасибо")
    assert await program.repo.get_kv("pending_files") == '["files/inbox/2_b.stl"]'
    assert program.bot.last().text.startswith("❓ Новая задача")  # the draft waits

    # Then a text: the new draft holds the file, and it waits no more.
    card = await program.drafted("И найди, кто сделает 10 ключей", [draft(title="Ключи")])
    second = await program.repo.get_task(card.task_id)
    assert second.attachments == ["files/inbox/2_b.stl"]
    assert await program.repo.get_kv("pending_files") == "[]"
    assert program.bot.last() is card
    assert_tidy(program)


async def test_story_a_voice_message_gets_one_line_and_no_model(
    program: Program,
) -> None:
    (line,) = await program.other("🎤 голосовое")

    assert line.text == (
        "Пока понимаю только текст. Напиши словами — или пришли файл вместе с задачей."
    )
    assert program.llm.requests == []
    assert program.bot.last() is line


async def test_story_old_commands_stop_and_continue_in_words(program: Program) -> None:
    await program.launched()
    program.script(END)
    await program.round()

    (tasks,) = await program.command("tasks")
    assert tasks.text == (
        "Команды больше не нужны — напиши словами: «что по задачам?»,"
        " «поставь задачу 3 на паузу»"
    )
    assert program.bot.last().kind == "board"  # one status, last
    assert len([c for _, c in program.bot.chat() if c.kind == "board"]) == 1

    await program.command("stop")
    assert program.board().split("\n")[1] == (
        "⏸ Агент остановлен — ничего не пишет и не отвечает"
    )
    assert program.bot.last().kind == "board"

    program.secretary([call("set_running", on=True, quote="продолжай")], [])
    await program.write("продолжай")
    assert await program.repo.is_stopped() is False
    assert "⏸" not in program.board()
    await program.assert_clean()


async def test_story_a_burst_is_read_as_one_and_survives_a_restart(
    program: Program,
) -> None:
    # Ruling R71: the owner writes one request as several short messages in a row.
    scheduled: list[object] = []
    program.app.logic.timer = lambda delay, job: scheduled.append(job)
    for text in ("Найди, кто напечатает", "5 корпусов из PETG"):
        assert await program.write(text) == []
    assert program.asked() == [] and len(scheduled) == 2

    program.app = program.build()  # the program stopped before the burst was read
    program.secretary([draft(title="Печать корпусов")], [])
    await start_up(program.app)

    (task,) = await program.repo.all_tasks()
    assert task.title == "Печать корпусов"
    assert task.source_text == ""  # several messages: the brief only (R38)
    content = program.asked()[0]["messages"][0]["content"]
    assert "1. Найди, кто напечатает" in content and "2. 5 корпусов из PETG" in content
    assert program.bot.last().text.startswith(f"❓ Новая задача #{task.id}")


ALISHER_PEER = 2002


def find_person(name: str) -> dict[str, Any]:
    block = call("find_person")
    block["input"] = {"name": name}
    return block


def sent_to(program: Program, peer: int) -> list[str]:
    return [text for to, text in program.telegram.sent if to == peer]


async def test_story_get_in_touch_with_a_person_from_the_contacts_by_name(
    program: Program,
) -> None:
    # Task 5.4 (the owner's check list): «свяжись с <имя из контактов>» → the draft's card
    # shows whom → «Запустить» → the message goes to him.
    program.telegram.contacts = [
        PeerInfo(ALISHER_PEER, "alisher_k", "Алишер Каримов", False, "+998901231234"),
        PeerInfo(2003, "dilshod_m", "Дилшод", False, "+998935550011"),
    ]
    program.telegram.peers["alisher_k"] = PeerInfo(
        ALISHER_PEER, "alisher_k", "Алишер Каримов", False
    )

    card = await program.drafted(
        "Узнай у Алишера, готов ли заказ",
        [find_person("Алишер")],
        [draft(title="Заказ", brief="Узнать у Алишера, готов ли заказ")],
        [call("take_person", task_id=1, matches=["m1"], quote="Алишера")],
    )

    assert "• Алишер Каримов, @alisher_k — напишу после запуска" in card.text
    waiting = [c for _, c in program.bot.chat() if c.text.startswith("❓")]
    assert waiting == [card]  # the draft's card is the clarification: no second card
    assert program.bot.last() is card
    launch = next(d for row in card.buttons for _, d in row if ":launch:" in d)
    assert await program.press(launch) == "Запущено"
    (conv,) = await program.repo.task_conversations(card.task_id)

    program.script([send(conv.id, "Здравствуйте! Подскажите, заказ уже готов?")], END)
    await program.round()
    await program.until(
        lambda: bool(sent_to(program, ALISHER_PEER)), timedelta(minutes=1)
    )

    (first,) = sent_to(program, ALISHER_PEER)
    assert first.startswith("Здравствуйте!") and "заказ уже готов" in first
    assert sent_to(program, 2003) == []
    assert program.telegram.searches == []  # never Telegram's global search
    await program.assert_clean()


async def test_story_nobody_of_them_then_the_owner_picks_him_in_telegrams_list(
    program: Program,
) -> None:
    # Task 5.4: two Алишеры, neither is the one; the owner picks him in Telegram's list.
    program.telegram.contacts = [
        PeerInfo(2003, "alisher_b", "Алишер Бобуров", False, "+998935550011"),
        PeerInfo(2004, None, "Алишер Турсунов", False, "+998977770000"),
    ]
    await program.drafted(
        "Узнай у Алишера, готов ли заказ",
        [find_person("Алишер")],
        [draft(title="Заказ", brief="Узнать у Алишера, готов ли заказ")],
        [call("take_person", task_id=1, matches=["m1", "m2"], quote="Алишера")],
    )
    ask = program.bot.last()
    assert ask.text.startswith("❓ Кого ты имел в виду?")

    await program.press(ask.buttons[-1][0][1])  # «Никто из них»

    assert program.bot.screen[program.msg_id(ask)].text == "Никто из них"
    offer = next(c for c in reversed(program.bot.sent) if c.buttons == [
        [("Выбрать из контактов", "users:1")]
    ])  # fmt: skip
    assert "пришли его контакт" in offer.text

    program.secretary([call("take_person", task_id=1, matches=["p1"], quote="")], [])
    await program.app.logic.layout.saw(program.bot.owner_says("Алишер Каримов"))
    await program.show(
        await program.app.logic.on_person(OWNER, "Алишер Каримов", "alisher_k", None)
    )
    await program.app.logic.layout.settle()

    assert await program.repo.get_kv("task_targets:1") == '["alisher_k"]'
    card = program.bot.last()
    assert card.text.startswith("❓ Новая задача #1")
    assert "• Алишер Каримов, @alisher_k — напишу после запуска" in card.text


# -- stage 4б: a site that wants a sign-in, signed in from the phone --


class RemoteTab:
    """The remote tab of the agent's browser, faked: it opens whatever it is given."""

    def __init__(self) -> None:
        self.started: list[str] = []

    async def start(
        self,
        url: str,
        on_frame: Callable[[bytes], None],
        on_refused: Callable[[], None],
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        self.started.append(url)

    async def pointer(self, kind: str, x: float, y: float) -> None: ...

    async def wheel(self, dy: float) -> None: ...

    async def text(self, s: str) -> None: ...

    async def key(self, k: str) -> None: ...

    async def back(self) -> None: ...

    async def focus(self) -> str:
        return "none"

    def url(self) -> str:
        return ""

    async def stop(self) -> None: ...


class QuickTunnel:
    name = "cloudflared"
    pid = None

    async def open(self, port: int) -> str:
        return "https://quiet-words.trycloudflare.com"

    async def close(self) -> None: ...


async def test_story_a_site_that_wants_a_sign_in_is_signed_in_from_the_phone(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    settings = Settings(
        owner=settings.owner,
        data_dir=settings.data_dir,
        remote=Remote(tunnel="cloudflared"),
    )
    olx = Wall("olx.uz", "https://www.olx.uz/account/", "login")
    search = FakeSearch(
        outcomes=[
            SearchOutcome(walls=[olx], notes=["OLX: телефон виден только после входа"]),
            SearchOutcome(
                candidates=[Candidate("PrintLab", "print_lab", "https://olx.uz/d/1", "")]
            ),
        ],
        repo=repo,
    )
    tab, servers = RemoteTab(), []

    def server() -> RemoteServer:
        servers.append(RemoteServer(0))
        return servers[-1]

    program = Program(
        repo, fake_clock, settings, FakeLLM(), searcher=search,
        remote_screen=lambda: tab, remote_tunnel=QuickTunnel, remote_server=server,
    )  # fmt: skip
    # A task without contacts; its agent searches, and the search meets OLX's sign-in.
    card = await program.drafted("Найди, кто напечатает 5 корпусов из PETG", [draft()])
    launch = next(data for row in card.buttons for _, data in row if ":launch:" in data)
    assert await program.press(launch) == "Запущено"
    program.script([call("research", what="мастерские 3D-печати PETG")], END)
    await program.round()

    offer = program.bot.last()
    assert offer.kind == "site_login"
    assert offer.text.startswith("❓ Сайт olx.uz показывает контакты только после входа.")

    # «Войти»: the link comes in a message of its own, only in its button.
    sign_in = next(data for row in offer.buttons for label, data in row if label == "Войти")
    assert await program.press(sign_in) == (
        "Открываю страницу входа: ссылка придёт через несколько секунд."
    )
    await program.app.site_logins.ready()
    await program.app.logic.layout.settle()
    assert program.bot.screen[program.msg_id(offer)].text == "Вход на olx.uz: ссылка ниже"
    link_card = next(c for c in reversed(program.bot.sent) if c.kind == "site_link")
    ((label, data),) = link_card.buttons[0]
    assert label == "Открыть страницу входа" and data.startswith("url:https://")
    assert tab.started == ["https://www.olx.uz/account/"]

    # On the phone he signs in and presses «Готово».
    address, fragment = data[4:].split("#")
    token = address.rsplit("/s/", 1)[1]
    done = json.dumps({"p": secrets.token_hex(16), "n": 1, "t": "done"}).encode()
    async with (
        aiohttp.ClientSession() as http,
        http.ws_connect(
            f"http://127.0.0.1:{servers[0].port}/s/{token}/ws",
            origin="https://quiet-words.trycloudflare.com",
        ) as ws,
    ):
        await ws.send_bytes(seal(RemoteKey.from_fragment(fragment), done))
        async for _ in ws:
            pass
    await program.app.site_logins.idle()
    assert program.bot.notes[-1] == "Вход на olx.uz сохранён."

    # The task's agent wakes with the news and searches again, on olx.uz now.
    program.script([call("research", what="мастерские 3D-печати PETG на olx.uz")], END)
    await program.round()

    assert "Владелец вошёл на сайт olx.uz" in program.context()
    assert [c.sites for c in search.calls] == [[], ["olx.uz"]]
    await program.assert_clean()
    await program.app.site_logins.close()


# -- Task 6.1 (R76): a task that spent its money asks for more with one button --


async def test_a_task_out_of_money_goes_on_only_after_the_button(
    program: Program,
) -> None:
    from tg_agent.brain.schemas import Usage

    await program.launched()
    task_id = program.task_id
    await program.repo.log_llm_call(
        None, "task", "m", Usage(1, 1), 3.0, "end_turn", task_id=task_id
    )

    await program.round()
    await program.round(timedelta(minutes=5))

    assert await program.wakes() == 0
    (card,) = [c for c in program.cards() if c.kind == "task_money"]
    assert card.text.startswith(f"❓ Задача #{task_id} «{TITLE}» потратила $3.00.")
    data = card.buttons[0][0][1]

    async def money_card() -> Card:
        """The card as it reads now (the layout may have moved it to the bottom)."""
        msg = await program.app.logic.layout.last_message("task_money", task_id)
        assert msg is not None
        return program.bot.screen[msg]

    # Words are an instruction, never money: the task still waits, the card stays.
    # A real step of the secretary (review M18): the instruction is taken and reaches the
    # task, and still raises no money.
    words = f"продолжи задачу {task_id}, дай ещё денег"
    program.secretary(
        [call("instruct_task", task_id=task_id, quote=words)],
        say("Передал."),
    )
    await program.write(words)
    answers = [
        item
        for _, request in program.llm.requests[-1:]
        for message in request["messages"]
        if message["role"] == "user" and not isinstance(message["content"], str)
        for item in message["content"]
        if item.get("type") == "tool_result"
    ]
    assert answers and not any(item.get("is_error") for item in answers), answers
    assert EventKind.OWNER_INSTRUCTION in [
        event.kind for event in await program.repo.open_events(task_id)
    ]
    await program.round(timedelta(minutes=1))
    assert await program.wakes() == 0
    assert await program.repo.get_kv(f"task_allowance:{task_id}") is None
    assert (await money_card()).buttons

    assert await program.press(data) == f"Задача #{task_id}: продолжаю, ещё $3"
    assert await program.repo.get_kv(f"task_allowance:{task_id}") == "6.0"
    assert (await money_card()).text == f"Задача #{task_id}: продолжаю, ещё $3"
    program.script([send(program.person, GREETING)], END)
    await program.round()
    assert await program.wakes() == 1

    assert await program.press(data) == "Уже решено"
    assert await program.press(f"mn:{task_id}:more:abcdef") == "Уже решено"
    assert await program.repo.get_kv(f"task_allowance:{task_id}") == "6.0"


async def test_an_unpressed_money_card_comes_again_the_next_day(
    program: Program,
) -> None:
    from tg_agent.brain.schemas import Usage

    await program.launched()
    task_id = program.task_id
    await program.repo.log_llm_call(
        None, "task", "m", Usage(1, 1), 3.0, "end_turn", task_id=task_id
    )

    def codes() -> list[str]:
        """The distinct buttons of the money cards sent so far, in order."""
        found = [
            c.buttons[0][0][1] for c in program.cards() if c.kind == "task_money"
        ]
        return list(dict.fromkeys(found))

    await program.round()
    await program.round(timedelta(hours=23))
    assert len(codes()) == 1
    await program.round(timedelta(hours=2))

    old, new = codes()
    assert old != new
    assert await program.press(old) == "Уже решено"
    assert await program.press(new) == f"Задача #{task_id}: продолжаю, ещё $3"
    assert await program.repo.get_kv(f"task_allowance:{task_id}") == "6.0"
    assert await program.wakes() == 0
