"""The layout of the owner's chat with the bot (`control/layout.py`, stage 5, R52–R56).

Each test drives the pieces the way the bot does: the owner's message is reported to the
layout (`saw`), the logic answers, its cards are shown, and `settle()` runs after the handler.
What is asserted is the chat as `FakeUI` shows it: which message is last, and what each reads.
"""

from __future__ import annotations

import logging
import random
from datetime import timedelta

import pytest

from tg_agent.agent.ops import TaskOps
from tg_agent.agent.runner import Runner
from tg_agent.brain.llm import FakeLLM
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import Card, CardGone, FakeUI
from tg_agent.control.cards import RESULT_MARK, STATUS_MARK, WAITING_MARK, result_card
from tg_agent.control.layout import ChatLayout
from tg_agent.control.logic import ControlLogic
from tg_agent.db import Repo
from tg_agent.gateway.base import PeerInfo
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import ApprovalKind, ApprovalStatus, ConvState, TaskStatus

OWNER = 111
TITLE = "3D-печать"
MOVED = "⬇ перенесено ниже"


_calls = iter(range(1, 100_000))


def draft_call(task_id: int | None = None, *, people: tuple[str, ...] = ()) -> dict:
    """The secretary's model drafts the task (or rewrites draft `task_id`)."""
    return {
        "type": "tool_use", "id": f"l{next(_calls)}", "name": "draft_task",
        "input": {
            "task_id": task_id, "title": TITLE, "brief": "Узнать цену",
            "people": list(people), "ceiling": None, "currency": "UZS",
            "share": ["name"], "missing": [], "uses_files": False,
        },
    }  # fmt: skip


def data_of(card: Card) -> list[str]:
    return [data for row in card.buttons for _, data in row]


class Chat:
    """The owner's bot on fakes, with the layout between the program and the bot."""

    def __init__(self, repo: Repo, clock: FakeClock, settings: Settings) -> None:
        self.repo, self.clock = repo, clock
        self.ui = FakeUI()
        self.layout = ChatLayout(repo, self.ui, clock=clock)
        llm, self.gateway = FakeLLM(), FakeGateway(clock)
        self.gateway.peers["print_lab"] = PeerInfo(1001, "print_lab", "Акмаль", False)
        self.gateway.peers["kley_shop"] = PeerInfo(1002, "kley_shop", "Бобур", False)
        self.runner = Runner(
            repo, llm, self.gateway, self.layout, clock, settings, random.Random(1),
            canary="ZX-layout-1", owner_secrets=[], inline=True,
        )  # fmt: skip
        self.ops = TaskOps(
            repo, self.gateway, clock, settings, self.runner, random.Random(1)
        )
        self.logic = ControlLogic(
            repo, self.ops, self.runner.approvals, llm, settings, clock, self.layout
        )
        self.llm = llm

    # -- the owner, as the bot's handlers carry him --

    async def write(self, text: str, reply_to: int | None = None) -> list[Card]:
        await self.layout.saw(self.ui.owner_says(text))
        cards = await self.logic.on_text(OWNER, text, reply_to, [])
        for card in cards:
            await self.logic.show(card)
        await self.layout.settle()
        return cards

    async def press(self, data: str) -> str:
        toast = await self.logic.on_callback(OWNER, data)
        await self.layout.settle()
        return toast

    # -- the chat --

    def last(self) -> Card:
        return self.ui.last()

    def last_id(self) -> int:
        return self.ui.chat()[-1][0]

    def ids_of(self, kind: str) -> list[int]:
        return [msg_id for msg_id, card in self.ui.chat() if card.kind == kind]

    def board_id(self) -> int:
        (msg_id,) = self.ids_of("board")
        return msg_id

    # -- the openings --

    def secretary(self, *steps: object) -> None:
        """Script the secretary's model for the owner's next message."""
        self.llm.steps["secretary"].extend(steps)

    async def drafted(self, *targets: str) -> Card:
        """The owner writes a task; the secretary drafts it and shows its card."""
        self.secretary([draft_call(people=targets)], [])
        assert await self.write("Узнай цену печати у @print_lab, 5 корпусов") == []
        card = next(c for c in reversed(self.ui.sent) if c.kind == "task")
        return card

    async def launched(self) -> int:
        card = await self.drafted("print_lab")
        launch = next(d for d in data_of(card) if ":launch:" in d)
        assert await self.press(launch) == "Запущено"
        return card.task_id

    async def person(self, task_id: int) -> int:
        (conv,) = await self.repo.task_conversations(task_id)
        await self.repo.set_state(conv.id, ConvState.CONTACTED)
        await self.repo.set_state(conv.id, ConvState.TALKING)
        return conv.id

    async def asks(self, task_id: int, conv_id: int | None, question: str) -> int:
        """The agent asks the owner; returns the approval's id."""
        task = await self.repo.get_task(task_id)
        _, is_error = await self.runner.approvals.ask(
            task, kind="question", question=question, conv_id=conv_id,
            suggested=["Беру"], amount=None, terms="",
        )  # fmt: skip
        assert not is_error
        approvals = [
            a for a in await self.repo.pending_approvals() if a.task_id == task_id
        ]
        return approvals[-1].id

    async def finished(self, task_id: int, result: str = "PrintLab: 240 000") -> int:
        """The agent reports; returns the result card's message id. The task stays in work
        until the owner ends it (R109)."""
        await self.repo.set_kv(f"task_result:{task_id}", result)
        await self.repo.set_kv(f"task_result_waits:{task_id}", "1")
        msg_id = await self.layout.send(result_card(task_id, TITLE, result))
        await self.repo.set_kv(f"task_card:{msg_id}", str(task_id))
        return msg_id


@pytest.fixture
def chat(repo: Repo, fake_clock: FakeClock, settings: Settings) -> Chat:
    return Chat(repo, fake_clock, settings)


# 1


async def test_nothing_waits_and_the_status_is_last_then_it_moves_below_the_answer(
    chat: Chat,
) -> None:
    await chat.launched()
    status = chat.board_id()
    assert chat.last_id() == status
    sent = len(chat.ui.sent)

    await chat.layout.settle()
    assert len(chat.ui.sent) == sent  # the status is last already: nothing is sent

    task_id = (await chat.repo.all_tasks())[0].id
    (card,) = await chat.write("уточни срок", reply_to=chat.ids_of("task")[0])
    assert card.text == "Передал агенту"

    assert status in chat.ui.deleted
    new = chat.board_id()
    assert chat.last_id() == new != status
    assert new in chat.ui.silent and chat.ui.pins[-1] == new
    assert await chat.repo.get_kv("board_msg_id") == str(new)
    assert chat.last().text.startswith(f"{STATUS_MARK} Статус задач")
    assert f"#{task_id} {TITLE} — в работе" in chat.last().text


# 2


async def test_a_question_stays_last_and_its_answer_settles_it(chat: Chat) -> None:
    task_id = await chat.launched()
    conv_id = await chat.person(task_id)
    approval_id = await chat.asks(task_id, conv_id, "Брать за 240 000?")
    await chat.layout.settle()
    question = chat.last_id()
    assert chat.last().kind == "approval" and chat.last().text.startswith(WAITING_MARK)
    status = chat.board_id()

    await chat.repo.update_conversation(conv_id, note="240 000, 3 дня")
    await chat.layout.settle()
    assert chat.last_id() == question  # the card stays the last message
    assert chat.board_id() == status  # the status was edited above it, not moved
    assert "240 000, 3 дня" in chat.ui.screen[status].text

    (answer,) = await chat.write("бери, если до пятницы", reply_to=question)
    assert answer.text == "Ответ передан агенту"
    settled = chat.ui.screen[question]
    assert settled.text == (
        f"Вопрос — {TITLE}: «Брать за 240 000?»\nОтвет: «бери, если до пятницы»"
    )
    assert settled.buttons == []
    assert chat.last().kind == "board"
    assert (await chat.repo.get_approval(approval_id)).status is ApprovalStatus.ANSWERED


# 3


async def test_the_newest_waiting_card_comes_back_down_and_its_reply_still_answers(
    chat: Chat,
) -> None:
    task_id = await chat.launched()
    conv_id = await chat.person(task_id)
    first = await chat.asks(task_id, conv_id, "Брать?")
    second = await chat.asks(task_id, None, "Искать ещё?")
    old = (await chat.repo.get_approval(second)).control_msg_id
    await chat.layout.notify("Сообщение не ушло")  # a third message after them

    await chat.layout.settle()

    new = (await chat.repo.get_approval(second)).control_msg_id
    assert old in chat.ui.deleted and new == chat.last_id() != old
    assert new in chat.ui.silent
    assert chat.last().text.startswith(f"{WAITING_MARK} Вопрос")
    assert (await chat.repo.get_approval(first)).control_msg_id in chat.ui.screen

    await chat.write("ищи ещё двоих", reply_to=new)
    assert (await chat.repo.get_approval(second)).status is ApprovalStatus.ANSWERED
    # The other card waits still: it is the last message now.
    assert chat.last_id() == (await chat.repo.get_approval(first)).control_msg_id


# 4


async def test_a_launch_edits_the_drafts_card_and_its_reply_is_an_instruction(
    chat: Chat,
) -> None:
    card = await chat.drafted("print_lab")
    assert card.text.startswith(f"{WAITING_MARK} Новая задача #{card.task_id}")
    draft_id = chat.last_id()
    sent = len(chat.ui.sent)
    launch = next(d for d in data_of(card) if ":launch:" in d)

    assert await chat.press(launch) == "Запущено"

    launched = chat.ui.screen[draft_id]
    assert launched.text == f"Задача #{card.task_id} «{TITLE}» запущена"
    assert launched.buttons == [[("Закрыть задачу", f"fn:{card.task_id}:ask:-")]]
    # No second message: only the status is new, and it is last.
    assert [c.kind for c in chat.ui.sent[sent:]] == ["board"]
    assert chat.last().kind == "board"
    (said,) = await chat.write("поторгуйся", reply_to=draft_id)
    assert said.text == "Передал агенту"


async def test_a_closed_draft_settles_and_a_rewritten_one_keeps_one_card(
    chat: Chat,
) -> None:
    card = await chat.drafted()
    draft_id = chat.last_id()
    cancel = next(d for d in data_of(card) if ":cancel:" in d)
    await chat.press(cancel)
    assert chat.ui.screen[draft_id].text == f"Задача #{card.task_id} «{TITLE}» закрыта"
    assert chat.ui.screen[draft_id].buttons == []

    other = await chat.drafted()
    other_id = chat.last_id()
    chat.secretary([draft_call(other.task_id, people=("print_lab",))], [])
    assert await chat.write("добавь @print_lab", reply_to=other_id) == []
    assert other_id not in chat.ui.screen  # the old card went: one card of a draft
    assert chat.last().text.startswith(f"{WAITING_MARK} Новая задача #{other.task_id}")
    assert "@print_lab" in chat.last().text
    drafts = [c for _, c in chat.ui.chat() if c.text.startswith(WAITING_MARK)]
    assert len(drafts) == 1


# 5


async def test_a_result_waits_for_finish_and_finish_settles_it(chat: Chat) -> None:
    task_id = await chat.launched()
    conv_id = await chat.person(task_id)
    result_id = await chat.finished(task_id)
    await chat.layout.settle()
    assert chat.last_id() == result_id
    assert chat.last().text.startswith(f"{RESULT_MARK} Готово — {TITLE}")

    toast = await chat.press(f"fn:{task_id}:ack:-")

    assert toast == f"Задача #{task_id} завершена"
    assert (
        chat.ui.screen[result_id].text
        == f"{RESULT_MARK} Готово — {TITLE}\nPrintLab: 240 000"
    )
    assert chat.ui.screen[result_id].buttons == []
    conv = await chat.repo.get_conversation(conv_id)
    assert conv.state is ConvState.CLOSED and conv.close_reason == "задача завершена"
    assert chat.gateway.sent == []  # no goodbye
    assert (await chat.repo.get_task(task_id)).status is TaskStatus.DONE
    assert chat.last().kind == "board"
    assert chat.last().text == f"{STATUS_MARK} Задач в работе нет"


async def test_a_reply_to_a_result_takes_the_task_up_and_the_card_stops_waiting(
    chat: Chat,
) -> None:
    task_id = await chat.launched()
    await chat.person(task_id)
    result_id = await chat.finished(task_id)
    await chat.layout.settle()

    (card,) = await chat.write("уточни срок", reply_to=result_id)

    assert card.text == f"Задача #{task_id} снова в работе: передал агенту"
    assert chat.ui.screen[result_id].buttons == []
    assert chat.last().kind == "board"
    assert f"#{task_id} {TITLE} — в работе" in chat.last().text


# 6


async def test_an_approval_that_expires_by_itself_says_it_was_not_needed(
    chat: Chat,
) -> None:
    task_id = await chat.launched()
    conv_id = await chat.person(task_id)
    await chat.asks(task_id, conv_id, "Брать?")
    await chat.layout.settle()
    question = chat.last_id()
    from tg_agent.agent import people

    await people.close(chat.repo, chat.gateway, conv_id, "не отвечает")
    await chat.layout.settle()

    assert chat.ui.screen[question].text == f"Вопрос — {TITLE} — не понадобилось"
    assert chat.ui.screen[question].buttons == []
    assert chat.last().kind == "board"


# 7


async def test_a_hand_back_card_and_a_notice_are_not_tracked(chat: Chat) -> None:
    task_id = await chat.launched()
    conv_id = await chat.person(task_id)
    await chat.repo.set_state(conv_id, ConvState.HANDED_TO_OWNER)
    conv = await chat.repo.get_conversation(conv_id)
    assert await chat.runner.approvals.send_hand_back(task_id, conv)
    hand_back = chat.last_id()
    assert not chat.last().text.startswith(WAITING_MARK)

    await chat.layout.settle()
    assert chat.last().kind == "board" and hand_back in chat.ui.screen

    await chat.layout.notify("Лимит денег на сегодня исчерпан")
    await chat.layout.settle()
    assert chat.last().kind == "board"

    approval_id = await chat.asks(task_id, None, "Искать ещё?")
    await chat.layout.notify("Ещё одно сообщение")
    await chat.layout.settle()
    assert chat.last_id() == (await chat.repo.get_approval(approval_id)).control_msg_id


# 8


async def test_a_delete_telegram_refuses_leaves_a_line_and_sends_the_new_one(
    chat: Chat, caplog: pytest.LogCaptureFixture
) -> None:
    await chat.launched()
    old = chat.board_id()
    chat.ui.undeletable.add(old)
    await chat.layout.notify("Сообщение")

    await chat.layout.settle()

    assert chat.ui.screen[old].text == MOVED and chat.ui.screen[old].buttons == []
    new = chat.ids_of("board")[-1]
    assert chat.last_id() == new != old
    assert await chat.repo.get_kv("board_msg_id") == str(new)

    # The old one cannot be edited either: logged, and the move happens all the same.
    async def refuses(msg_id: int, card: Card) -> None:
        raise CardGone("Bad Request: message can't be edited")

    chat.ui.undeletable.add(new)
    chat.ui.edit = refuses  # type: ignore[method-assign]
    await chat.layout.notify("Ещё")
    with caplog.at_level(logging.INFO):
        await chat.layout.settle()
    assert chat.last().kind == "board" and chat.last_id() not in (old, new)


# 9


async def test_settle_twice_changes_nothing_and_a_failing_ui_does_not_raise(
    chat: Chat,
) -> None:
    task_id = await chat.launched()
    await chat.asks(task_id, None, "Искать ещё?")
    await chat.layout.notify("Сообщение")
    await chat.layout.settle()
    before = (len(chat.ui.sent), len(chat.ui.edits), len(chat.ui.deleted))
    await chat.layout.settle()
    assert (len(chat.ui.sent), len(chat.ui.edits), len(chat.ui.deleted)) == before

    async def offline(msg_id: int) -> None:
        raise ConnectionError("network is down")

    chat.ui.delete = offline  # type: ignore[method-assign]
    await chat.layout.notify("Ещё")
    await chat.layout.settle()  # does not raise
    del chat.ui.delete
    await chat.layout.settle()  # and the next call does the move
    assert chat.last().kind == "approval"


# 10


async def test_the_markers(chat: Chat) -> None:
    card = await chat.drafted("print_lab")
    assert card.text[0] == WAITING_MARK
    launch = next(d for d in data_of(card) if ":launch:" in d)
    await chat.press(launch)
    assert chat.last().text[0] == STATUS_MARK
    assert chat.ui.screen[chat.ids_of("task")[0]].text[0] not in (
        WAITING_MARK, STATUS_MARK, RESULT_MARK,
    )  # fmt: skip
    task_id = card.task_id
    conv_id = await chat.person(task_id)
    await chat.asks(task_id, conv_id, "Брать?")
    await chat.layout.settle()
    assert chat.last().text[0] == WAITING_MARK
    question = chat.last_id()
    await chat.write("бери", reply_to=question)
    assert chat.ui.screen[question].text[0] != WAITING_MARK
    await chat.finished(task_id)
    await chat.layout.settle()
    assert chat.last().text[0] == RESULT_MARK


# 11


async def test_the_close_confirmation_waits_and_settles_in_place(chat: Chat) -> None:
    task_id = await chat.launched()
    ask = f"fn:{task_id}:ask:-"

    assert await chat.press(ask) == ""
    confirm = chat.last_id()
    assert chat.last().kind == "finish_ask"
    assert chat.last().text.startswith(
        f"{WAITING_MARK} Закрыть задачу #{task_id} «{TITLE}»?"
    )
    no = next(d for d in data_of(chat.last()) if ":no:" in d)
    await chat.layout.notify("Сообщение")
    await chat.layout.settle()
    confirm = chat.last_id()  # moved down: it waits
    assert chat.last().kind == "finish_ask"

    await chat.press(no)
    assert chat.ui.screen[confirm].text == f"Задача #{task_id}: не закрываю"
    assert chat.last().kind == "board"

    await chat.press(ask)
    first = chat.last_id()
    await chat.press(ask)  # a second one settles the first
    assert chat.ui.screen[first].text == (
        f"Закрыть задачу #{task_id} «{TITLE}»? — не понадобилось"
    )
    second = chat.last_id()
    yes = next(d for d in data_of(chat.last()) if ":yes:" in d)
    notes = len(chat.ui.notes)
    toast = await chat.press(yes)
    assert toast == f"Задача #{task_id} закрыта"
    assert chat.ui.screen[second].text == toast and chat.ui.screen[second].buttons == []
    assert len(chat.ui.notes) == notes  # no separate notice
    assert chat.last().kind == "board"


async def test_a_close_confirmation_unanswered_for_a_day_is_settled(chat: Chat) -> None:
    task_id = await chat.launched()
    await chat.press(f"fn:{task_id}:ask:-")
    confirm = chat.last_id()
    chat.clock.advance(timedelta(hours=24, minutes=1))

    await chat.layout.settle()

    assert chat.ui.screen[confirm].text == (
        f"Закрыть задачу #{task_id} «{TITLE}»? — не понадобилось"
    )
    assert chat.last().kind == "board"


# 12


async def test_a_card_without_a_row_of_its_own_is_released_by_its_owner(
    chat: Chat,
) -> None:
    await chat.launched()
    login = Card(
        f"{WAITING_MARK} Войти на olx.uz\nНажми, когда войдёшь",
        [[("Вошёл", "lg:0:done:-")]],
        kind="site_login",
        key="olx.uz",
    )
    msg_id = await chat.layout.send(login)
    await chat.layout.settle()
    assert chat.last_id() == msg_id
    await chat.layout.notify("Сообщение")
    await chat.layout.settle()
    msg_id = chat.last_id()
    assert chat.last().kind == "site_login"

    assert await chat.layout.release(
        "site_login", "olx.uz", Card("Вход на olx.uz — готово")
    )
    assert chat.ui.screen[msg_id].text == "Вход на olx.uz — готово"
    await chat.layout.settle()
    assert chat.last().kind == "board"
    assert not await chat.layout.release("site_login", "olx.uz", None)

    again = await chat.layout.send(login)
    chat.clock.advance(timedelta(hours=25))
    await chat.layout.settle()
    assert chat.ui.screen[again].text == "Войти на olx.uz — не понадобилось"
    assert chat.last().kind == "board"


async def draft_card_of(chat: Chat, title: str) -> tuple[int, int]:
    """A draft and its card, sent as the logic sends one: (task id, message id)."""
    task_id = await chat.repo.create_task(title, "Узнать цену", {}, None, "UZS", [], [])
    card = Card(f"{WAITING_MARK} {title}", [[("Запустить", "x")]], task_id=task_id)
    card.kind = "task"
    return task_id, await chat.logic.show(card)


async def test_a_replaced_card_is_sent_anew_as_the_newest_and_its_id_follows(
    chat: Chat,
) -> None:
    task_id, old = await draft_card_of(chat, "Печать")
    _, other = await draft_card_of(chat, "Ключи")
    await chat.layout.settle()
    assert chat.last_id() == other

    new_card = Card(f"{WAITING_MARK} Печать, 10 штук", task_id=task_id, kind="task")
    new = await chat.layout.replace("task", task_id, new_card)

    assert new is not None and old in chat.ui.deleted
    assert chat.ui.screen[new].text == f"{WAITING_MARK} Печать, 10 штук"
    assert new not in chat.ui.silent  # what the owner just changed is news
    assert await chat.repo.get_kv(f"task_card:{new}") == str(task_id)
    assert await chat.repo.get_kv(f"task_card:{old}") == str(task_id)  # an old id still finds it
    await chat.layout.settle()
    assert chat.last_id() == new  # the newest waiting card: it stays last
    assert other in chat.ui.screen
    assert await chat.layout.replace("task", 999, new_card) is None  # nothing tracked


async def test_a_replaced_card_telegram_will_not_delete_leaves_a_line(
    chat: Chat,
) -> None:
    task_id, old = await draft_card_of(chat, "Печать")
    chat.ui.undeletable.add(old)

    new = await chat.layout.replace(
        "task", task_id, Card("Печать, 10 штук", task_id=task_id, kind="task")
    )

    assert chat.ui.screen[old].text == MOVED and new != old


async def test_a_card_redrawn_in_place_keeps_its_message(chat: Chat) -> None:
    task_id, msg_id = await draft_card_of(chat, "Печать")
    card = Card(f"{WAITING_MARK} Печать, файлы: k.stl", task_id=task_id, kind="task")

    assert await chat.layout.update("task", task_id, card)
    edits = len(chat.ui.edits)
    assert await chat.layout.update("task", task_id, card)  # the same: no edit

    assert chat.ui.screen[msg_id].text == card.text and len(chat.ui.edits) == edits
    assert not await chat.layout.update("task", 999, card)
    # A move later sends the card as it reads now.
    await chat.layout.notify("Сообщение")
    await chat.layout.settle()
    assert chat.last().text == card.text


async def test_an_old_status_telegram_will_not_delete_is_unpinned(chat: Chat) -> None:
    await chat.launched()
    old = chat.board_id()
    chat.ui.undeletable.add(old)
    await chat.layout.notify("Сообщение")

    await chat.layout.settle()

    assert chat.ui.unpins == [old]
    assert chat.ui.pins[-1] == chat.ids_of("board")[-1] != old


async def test_typing_reaches_the_bot(chat: Chat) -> None:
    await chat.layout.typing()
    assert chat.ui.typings == 1


async def test_an_approval_card_carries_the_marker_but_a_hand_back_does_not(
    chat: Chat,
) -> None:
    task_id = await chat.launched()
    conv_id = await chat.person(task_id)
    task = await chat.repo.get_task(task_id)
    await chat.runner.approvals.ask(
        task, kind="phone", question="Просит телефон", conv_id=conv_id,
        suggested=[], amount=None, terms="",
    )  # fmt: skip
    await chat.layout.settle()
    assert chat.last().text.startswith(
        f"{WAITING_MARK} Дать твой телефон? · @print_lab"
    )
    (approval,) = await chat.repo.pending_approvals(conv_id)
    assert approval.kind is ApprovalKind.SHARE_PHONE
    phone = chat.last_id()
    await chat.press(f"ap:{approval.id}:no:{approval.nonce}")
    assert chat.ui.screen[phone].text == "Телефон · @print_lab: не дали"


# -- the review of Task 5.1 (Part G of Task 5.3) --


async def test_a_reply_to_a_moved_questions_old_id_answers_it(chat: Chat) -> None:
    # I1: the Scheduler's settle runs between the bot's `saw` and the logic's lookup.
    task_id = await chat.launched()
    conv_id = await chat.person(task_id)
    approval_id = await chat.asks(task_id, conv_id, "Брать?")
    await chat.layout.settle()
    old = chat.last_id()

    await chat.layout.saw(chat.ui.owner_says("бери"))
    await chat.layout.settle()  # the question moves below the owner's reply
    assert old in chat.ui.deleted
    (answer,) = await chat.logic.on_text(OWNER, "бери", old, [])

    assert answer.text == "Ответ передан агенту"
    approval = await chat.repo.get_approval(approval_id)
    assert (approval.status, approval.answer) == (ApprovalStatus.ANSWERED, "бери")
    assert [t.id for t in await chat.repo.all_tasks()] == [task_id]  # no new draft


async def test_a_reply_to_the_moved_line_of_a_question_answers_it(chat: Chat) -> None:
    task_id = await chat.launched()
    approval_id = await chat.asks(task_id, None, "Искать ещё?")
    await chat.layout.settle()
    old = chat.last_id()
    chat.ui.undeletable.add(old)  # older than 48 hours
    await chat.layout.notify("Сообщение")
    await chat.layout.settle()
    assert chat.ui.screen[old].text == MOVED

    await chat.write("ищи ещё двоих", reply_to=old)

    approval = await chat.repo.get_approval(approval_id)
    assert approval.status is ApprovalStatus.ANSWERED


async def test_a_reply_to_a_moved_drafts_old_id_or_new_id_reaches_the_draft(
    chat: Chat,
) -> None:
    card = await chat.drafted()
    old = chat.last_id()
    await chat.layout.notify("Сообщение")
    await chat.layout.settle()
    new = chat.last_id()
    assert old in chat.ui.deleted and new != old

    for reply_to in (old, new):  # right after the move: no later settle healed it
        chat.secretary([{"type": "text", "text": "Понял."}])
        await chat.layout.saw(chat.ui.owner_says("а сроки?"))
        await chat.logic.on_text(OWNER, "а сроки?", reply_to, [])
        asked = [r for p, r in chat.llm.requests if p == "secretary"][-1]
        content = asked["messages"][0]["content"]
        assert f"это ответ на карточку черновика #{card.task_id}" in content


async def test_two_drafts_and_the_first_launched_leave_the_second_last(
    chat: Chat,
) -> None:
    # I2, the owner's check list: no status message existed before the launch.
    chat.secretary(
        [draft_call(people=("print_lab",)), draft_call(people=("kley_shop",))], []
    )
    await chat.write("Узнай цену печати у @print_lab. И цену клея у @kley_shop")
    first, second = [c for c in chat.ui.sent if c.kind == "task"]
    launch = next(d for d in data_of(first) if ":launch:" in d)

    assert await chat.press(launch) == "Запущено"

    assert chat.last().text.startswith(f"{WAITING_MARK} Новая задача #{second.task_id}")
    assert len(chat.ids_of("board")) == 1


async def test_a_status_the_owner_deleted_comes_back_above_the_waiting_card(
    chat: Chat,
) -> None:
    task_id = await chat.launched()
    await chat.asks(task_id, None, "Искать ещё?")
    await chat.layout.settle()
    question = chat.last_id()
    board = chat.board_id()
    chat.ui.deleted.append(board)  # the owner deleted it: it cannot be edited
    chat.ui.screen.pop(board)
    await chat.repo.set_task_notes(task_id, headline="Ищу дальше")

    await chat.layout.settle()

    assert chat.last().kind == "approval"
    assert chat.ids_of("board") and chat.last_id() != question  # moved below the status


async def test_a_launch_that_races_a_settle_shows_one_launched_card_with_its_notes(
    chat: Chat,
) -> None:
    # M1: a settle between the launch and the release settles the card without notes.
    chat.secretary([draft_call(people=("print_lab", "ghost_shop"))], [])
    await chat.write("Узнай цену печати у @print_lab и @ghost_shop")
    card = next(c for c in reversed(chat.ui.sent) if c.kind == "task")
    real = chat.ops.launch

    async def launch_then_settle(task_id: int) -> tuple[bool, list[str]]:
        outcome = await real(task_id)
        await chat.layout.settle()
        return outcome

    chat.ops.launch = launch_then_settle  # type: ignore[method-assign]
    launch = next(d for d in data_of(card) if ":launch:" in d)

    assert await chat.press(launch) == "Запущено"

    shown = [c.text for _, c in chat.ui.chat() if "запущена" in c.text]
    assert shown == [
        f"Задача #{card.task_id} «{TITLE}» запущена\n@ghost_shop: не найден — проверь имя"
    ]


async def test_a_result_of_a_task_paused_meanwhile_still_waits(chat: Chat) -> None:
    # M3, R109: the owner paused the task while the card was on its way. Its result still
    # waits: «Завершить» ends a paused task too.
    task_id = await chat.launched()
    result_id = await chat.finished(task_id)
    await chat.repo.set_task_status(task_id, TaskStatus.PAUSED)

    await chat.layout.settle()

    assert chat.ui.screen[result_id].buttons != []
    assert await chat.ops.acknowledge(task_id) == f"Задача #{task_id} завершена"
    await chat.layout.settle()
    assert chat.ui.screen[result_id].buttons == []


async def test_a_result_of_a_closed_task_stops_waiting(chat: Chat) -> None:
    task_id = await chat.launched()
    result_id = await chat.finished(task_id)
    await chat.layout.settle()
    assert chat.last_id() == result_id

    await chat.ops.close(task_id)
    await chat.layout.settle()

    assert chat.ui.screen[result_id].buttons == []
    assert chat.last().kind == "board"


async def test_a_card_that_waited_four_hours_comes_back_with_sound(chat: Chat) -> None:
    # R62: the reminder is the card itself.
    task_id = await chat.launched()
    approval_id = await chat.asks(task_id, None, "Искать ещё?")
    await chat.layout.settle()
    old = chat.last_id()
    await chat.layout.notify("Сообщение")  # something came after it, then settle skipped
    chat.clock.advance(timedelta(hours=4))

    assert await chat.layout.resend_newest() is True

    new = chat.last_id()
    assert old in chat.ui.deleted and new != old
    assert new not in chat.ui.silent  # with sound
    assert chat.last().text.startswith(f"{WAITING_MARK} Вопрос")
    assert (await chat.repo.get_approval(approval_id)).control_msg_id == new
    await chat.write("ищи ещё двоих", reply_to=new)
    assert (await chat.repo.get_approval(approval_id)).status is ApprovalStatus.ANSWERED


async def test_a_card_that_waited_two_hours_is_not_sent_again(chat: Chat) -> None:
    task_id = await chat.launched()
    await chat.asks(task_id, None, "Искать ещё?")
    await chat.layout.settle()
    sent = len(chat.ui.sent)
    chat.clock.advance(timedelta(hours=2))

    assert await chat.layout.resend_newest() is False
    assert len(chat.ui.sent) == sent


async def test_nothing_waits_and_nothing_is_sent_again(chat: Chat) -> None:
    await chat.launched()
    await chat.layout.settle()
    sent = len(chat.ui.sent)
    chat.clock.advance(timedelta(hours=10))

    assert await chat.layout.resend_newest() is False
    assert len(chat.ui.sent) == sent


async def test_a_reminder_that_fails_raises_so_it_is_tried_again(chat: Chat) -> None:
    task_id = await chat.launched()
    await chat.asks(task_id, None, "Искать ещё?")
    await chat.layout.settle()
    chat.clock.advance(timedelta(hours=4))

    async def offline(card: Card, *, silent: bool = False) -> int:
        raise ConnectionError("network is down")

    chat.ui.send = offline  # type: ignore[method-assign]
    with pytest.raises(ConnectionError):
        await chat.layout.resend_newest()
    del chat.ui.send
    assert await chat.layout.resend_newest() is True


async def test_an_older_message_never_moves_the_tail_back(chat: Chat) -> None:
    await chat.launched()
    tail = await chat.repo.get_kv("chat_tail")
    await chat.layout.saw(1)  # an update that arrived late
    assert await chat.repo.get_kv("chat_tail") == tail != "1"
