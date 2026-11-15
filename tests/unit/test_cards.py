from __future__ import annotations

import ast
import re
from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import tg_agent
from tg_agent.control.base import Card
from tg_agent.control.cards import (
    approval_card,
    board_card,
    conv_stage,
    draft_card,
    help_text,
    launched_card,
    masked,
    nobody_card,
    not_needed,
    person_label,
    pick_person_card,
    picked_card,
    result_card,
    settled_approval_card,
    unmarked,
)
from tg_agent.models import (
    Approval,
    ApprovalKind,
    ApprovalStatus,
    Contact,
    Conversation,
    ConvState,
    Task,
    TaskStatus,
)

T0 = datetime(2026, 11, 10, 10, 0, tzinfo=ZoneInfo("Asia/Tashkent"))
BIG = 9_999_999


# The five kinds of card the agent core makes.
CORE_KINDS = [
    ApprovalKind.QUESTION,
    ApprovalKind.CONFIRM_ORDER,
    ApprovalKind.SHARE_PHONE,
    ApprovalKind.SHARE_ADDRESS,
    ApprovalKind.HANDOFF,
]


def approval(kind: ApprovalKind, **payload: object) -> Approval:
    return Approval(
        id=BIG, task_id=BIG, conversation_id=BIG,
        kind=kind, payload=dict(payload), status=ApprovalStatus.PENDING, nonce="abcdef",
        control_msg_id=None, answer=None, created_at=T0, resolved_at=None,
    )  # fmt: skip


CONTACT = Contact(
    id=1, peer_id=1001, username="print_lab", display_name="Акмаль", is_bot=False,
    language="ru", card={}, do_not_contact=False, updated_at=T0,
)  # fmt: skip
CONV = Conversation(
    id=BIG, task_id=BIG, contact_id=1, state=ConvState.WAITING_OWNER, prev_state=None,
    facts={"price": "240 000 сум"}, summary="", language="ru", offtopic_strikes=0,
    agent_messages=2, followups_sent=0, first_sent=True, paused_until=None, last_in_at=T0,
    last_out_at=T0, created_at=T0, updated_at=T0,
)  # fmt: skip


def choices(card) -> list[str]:
    return [
        data.split(":")[2]
        for row in card.buttons
        for _, data in row
        if data.startswith("ap:")
    ]


def all_cards():
    draft = replace(TASK, id=BIG, ceiling=300_000, allowed=["name"])
    yield draft_card(draft, ["print_lab"], [], "abcdef")
    for kind in CORE_KINDS:
        yield approval_card(approval(kind, text="?"), CONV, CONTACT)


def test_callback_data_under_64_bytes() -> None:
    for card in all_cards():
        for row in card.buttons:
            for _, data in row:
                if not data.startswith("url:"):
                    assert len(data.encode("utf-8")) <= 64, data


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ApprovalKind.SHARE_PHONE, ["yes", "no"]),
        (ApprovalKind.SHARE_ADDRESS, ["yes", "no"]),
        (ApprovalKind.CONFIRM_ORDER, ["yes", "no", "self"]),
        (ApprovalKind.QUESTION, ["self"]),
        (ApprovalKind.HANDOFF, ["back"]),
    ],
)
def test_approval_buttons_per_kind(kind: ApprovalKind, expected: list[str]) -> None:
    card = approval_card(approval(kind, text="?"), CONV, CONTACT)
    assert choices(card) == expected
    assert all(
        data.endswith(":abcdef")
        for row in card.buttons
        for _, data in row
        if data.startswith("ap:")
    )


def test_handoff_card_opens_the_chat() -> None:
    card = approval_card(
        approval(ApprovalKind.HANDOFF, text="Не хочет с ИИ"), CONV, CONTACT
    )
    urls = [data for row in card.buttons for _, data in row if data.startswith("url:")]
    assert urls == ["url:https://t.me/print_lab"]


def test_a_card_may_carry_its_own_buttons() -> None:
    card = approval_card(
        approval(
            ApprovalKind.QUESTION, text="Лимит", buttons=[[["Продолжить", "more"]]]
        ),
        CONV,
        CONTACT,
    )
    assert [data.split(":")[2] for row in card.buttons for _, data in row] == ["more"]


def test_card_names_the_counterpart_and_quotes_them_as_such() -> None:
    card = approval_card(
        approval(
            ApprovalKind.QUESTION, text="Что ответить?", last_incoming="Скиньте номер"
        ),
        CONV,
        CONTACT,
    )
    assert "@print_lab" in card.text
    assert "От собеседника: «Скиньте номер»" in card.text
    assert "ответь на эту карточку" in card.text.lower()


def bolded(card) -> list[str]:
    return [card.text[start:end] for start, end in card.bold]


def test_draft_card_is_short_with_bold_headings() -> None:
    draft = replace(
        TASK, id=7, title="Цена печати", ceiling=300_000, allowed=["name"],
        goal="Узнать у @print_lab цену печати 5 корпусов из PETG", status=TaskStatus.DRAFT,
    )  # fmt: skip
    card = draft_card(draft, ["print_lab"], [], "abcdef")
    assert card.text == (
        "❓ Новая задача #7 — проверь и запусти\n\n"
        "1. Задача\n"
        "Цена печати\n\n"
        "2. Описание задачи\n"
        "Узнать у @print_lab цену печати 5 корпусов из PETG\n"
        "Бюджет до 300 000 сум — агенту не виден\n\n"
        "3. Делюсь\n"
        "имя\n\n"
        "4. Контакты\n"
        "• @print_lab — напишу после запуска\n\n"
        "5. Нужно уточнить\n"
        "—\n\n"
        "Поправить — напиши, что изменить: «добавь @username», «нужно 10 штук»."
    )
    assert bolded(card) == [
        "1. Задача",
        "2. Описание задачи",
        "3. Делюсь",
        "4. Контакты",
        "5. Нужно уточнить",
    ]
    assert [data for row in card.buttons for _, data in row] == [
        "tk:7:launch:abcdef",
        "tk:7:cancel:abcdef",
    ]
    assert [label for row in card.buttons for label, _ in row] == [
        "Запустить",
        "Закрыть задачу",
    ]
    assert card.task_id == 7


def test_draft_card_without_targets_finds_people_itself_and_can_launch() -> None:
    # Ruling R26: a task whose owner named nobody launches at once; its agent finds people.
    draft = replace(
        TASK, id=7, title="Ключи", goal="Узнать номер основателя Bukhara Artisans",
        allowed=["name"], status=TaskStatus.DRAFT,
    )  # fmt: skip
    card = draft_card(draft, [], ["какие ключи"], "abcdef")
    assert (
        "4. Контакты\nнайду сам\n\n5. Нужно уточнить\n• какие ключи\n\n"
    ) in card.text
    assert "кому писать" not in card.text and "@username реплаем" not in card.text
    assert "5. Нужно уточнить" in bolded(card) and "Бюджет" not in card.text
    assert [data for row in card.buttons for _, data in row] == [
        "tk:7:launch:abcdef",
        "tk:7:cancel:abcdef",
    ]
    assert [label for row in card.buttons for label, _ in row] == [
        "Запустить",
        "Закрыть задачу",
    ]


def test_draft_card_without_targets_and_nothing_missing() -> None:
    draft = replace(TASK, id=7, title="Ключи", goal="Сделать 10 ключей")
    card = draft_card(draft, [], [], "abcdef")
    assert "Файлы" not in card.text
    assert "4. Контакты\nнайду сам\n\n5. Нужно уточнить\n—\n\n" in card.text
    assert "Запустить" in [label for row in card.buttons for label, _ in row]


def test_a_draft_card_shows_a_person_taken_by_name_with_his_phone_masked() -> None:
    # Task 5.4: «Алишер Каримов, +998 90 ••• 12 34» — the card is the clarification.
    draft = replace(TASK, id=7, title="Заказ", goal="Узнать, готов ли заказ")
    names = {"+998901231234": "Алишер Каримов", "alisher_t": "Алишер Турсунов"}
    card = draft_card(
        draft, ["+998901231234", "alisher_t", "print_lab"], [], "abcdef", names
    )
    assert (
        "4. Контакты\n"
        "• Алишер Каримов, +998 90 ••• 12 34 — напишу после запуска\n"
        "• Алишер Турсунов, @alisher_t — напишу после запуска\n"
        "• @print_lab — напишу после запуска\n\n"
    ) in card.text
    assert "123" not in card.text.split("4. Контакты")[1].split("5.")[0]


@pytest.mark.parametrize(
    ("contact", "shown"),
    [
        ("+998901231234", "+998 90 ••• 12 34"),
        ("+79161234567", "+7916 ••• 45 67"),
        ("alisher_k", "@alisher_k"),
    ],
)
def test_a_phone_is_shown_with_its_middle_hidden(contact: str, shown: str) -> None:
    assert masked(contact) == shown


def test_a_persons_label_is_his_name_flat_and_cut_and_his_contact() -> None:
    assert person_label("Алишер\nКаримов", "+998901231234") == (
        "Алишер Каримов, +998 90 ••• 12 34"
    )
    assert person_label("", "alisher_k") == "@alisher_k"
    assert person_label("Тихий", None) == "Тихий"
    assert len(person_label("Я" * 100, "alisher_k")) < 60


def test_the_card_whom_did_you_mean_has_a_button_each_and_nobody() -> None:
    card = pick_person_card(
        7, "Заказ", ["Алишер Каримов, @alisher_k", "Алишер Бобуров, +998 93 ••• 00 11"],
        "abcdef",
    )  # fmt: skip
    assert card.text.startswith("❓ Кого ты имел в виду?")
    assert "#7 «Заказ»" in card.text
    assert card.buttons == [
        [("Алишер Каримов, @alisher_k", "pp:7:0:abcdef")],
        [("Алишер Бобуров, +998 93 ••• 00 11", "pp:7:1:abcdef")],
        [("Никто из них", "pp:7:none:abcdef")],
    ]
    assert (card.kind, card.key, card.task_id) == ("pick_person", "7", 7)
    assert picked_card("Алишер Каримов").text == "Алишер Каримов — выбран"
    assert nobody_card().text == "Никто из них"
    for settled in (picked_card("x"), nobody_card()):
        assert settled.buttons == [] and not settled.text.startswith("❓")


def test_a_draft_card_names_the_files_of_its_task() -> None:
    draft = replace(TASK, id=7, attachments=["data/files/inbox/43_m.stl", "x/7_ph.jpg"])
    card = draft_card(draft, [], [], "abcdef")
    assert "2. Описание задачи\nУзнать цену\nФайлы: m.stl, ph.jpg\n\n" in card.text


def test_the_board_has_a_close_button_for_every_task() -> None:
    # Саид, 2026-11-10: «Кнопка закрыть задачу для каждой задачи».
    other = Task(**{**TASK.__dict__, "id": 2, "title": "Ключи"})
    card = board_card([TASK, other], {1: [], 2: []}, T0)
    assert card.buttons == [
        [("Закрыть #1 3D-печать", "fn:1:ask:-")],
        [("Закрыть #2 Ключи", "fn:2:ask:-")],
    ]


def test_board_headings_are_bold() -> None:
    card = board_card([TASK], {1: []}, T0)
    assert bolded(card) == ["Статус задач", "#1 3D-печать — в работе"]


def test_help_says_how_to_talk_to_the_bot_in_words() -> None:
    text = help_text(time(9), 5)
    for said in (
        "обычными словами",
        "«Запустить»",  # a new task and what happens next
        "агент после запуска сам найдёт",  # no contacts
        "«искать можно и на OLX»",  # a correction in words
        "«по задаче 3: спроси ещё про доставку»",  # an instruction
        "«что по задачам?»",
        "«что ответил @username?»",
        "«поставь задачу 3 на паузу»",
        "«Закрыть #N»",
        "📋",
        "❓",
        "✅",
        "«Завершить»",
        "/stop",
        "«Продолжить работу»",
        "«продолжай»",
        "с 09:00",  # still true: first messages to new people
        "не больше 5 новых людей в день",
        "Файл",
    ):
        assert said in text, said
    assert "/help" not in text and "/start" not in text


def test_help_names_the_owners_own_hours_and_quota() -> None:
    text = help_text(time(8, 30), 3)
    assert "с 08:30" in text and "не больше 3 новых людей в день" in text


# Every command but these three is gone (ruling R64): no text the owner can see names one.
_REMOVED = re.compile(r"(?<![\w/.])/(tasks|task|pause|resume_all|resume|done|cost)(?![\w/.])")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                nodes.add(id(body[0].value))
    return nodes


def test_no_text_of_the_package_names_a_removed_command() -> None:
    root = Path(tg_agent.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docs
                and _REMOVED.search(node.value)
            ):
                found.append(f"{path.relative_to(root)}:{node.lineno}: {node.value!r}")
    assert found == []


def test_the_scan_would_find_a_removed_command() -> None:
    assert _REMOVED.search("Продолжить: /resume 3")
    assert _REMOVED.search("/tasks")
    assert not _REMOVED.search("brain/prompts/task.md") and not _REMOVED.search("/stop")


# --- what the owner sees of each counterpart: the status board, /tasks, /task N --------------

TASK = Task(
    id=1, title="3D-печать", goal="Узнать цену", item={}, ceiling=None, currency="UZS",
    allowed=[], attachments=[], owner_notes="", status=TaskStatus.ACTIVE, card_msg_id=None,
    created_at=T0, updated_at=T0,
)  # fmt: skip
OTHER = replace(CONTACT, id=2, peer_id=1002, username="3d_tash", display_name="Бобур")


def talking(*, last_out: datetime | None, last_in: datetime | None, **changes: object):
    fields: dict[str, object] = {"facts": {}, **changes}
    return replace(
        CONV,
        state=ConvState.TALKING,
        last_out_at=last_out,
        last_in_at=last_in,
        **fields,
    )


def test_board_tells_waiting_for_a_reply_from_a_reply_received() -> None:
    waiting = talking(last_out=T0, last_in=None)
    answered = talking(
        last_out=T0, last_in=T0 + timedelta(minutes=5), facts={"price": "240 000 сум"}
    )
    text = board_card(
        [TASK], {1: [(waiting, CONTACT), (answered, OTHER)]}, T0 + timedelta(hours=1)
    ).text
    assert "#1 3D-печать — в работе" in text
    assert "• @print_lab — жду ответа с 10:00" in text
    # The old engine's facts are not shown (Task 3.5): the agent's note is.
    assert "• @3d_tash — ответил в 10:05\n" in text + "\n"


def test_board_dates_a_wait_that_began_on_an_earlier_day() -> None:
    waiting = talking(last_out=T0 - timedelta(hours=15, minutes=30), last_in=None)
    text = board_card([TASK], {1: [(waiting, CONTACT)]}, T0).text
    assert "жду ответа с 09.11 18:30" in text


def test_board_without_tasks_says_so() -> None:
    assert board_card([], {}, T0).text == "📋 Задач в работе нет"


@pytest.mark.parametrize("state", list(ConvState))
def test_every_conversation_state_reads_in_russian(state: ConvState) -> None:
    label = conv_stage(replace(CONV, state=state, last_in_at=None, last_out_at=None))
    assert re.search("[а-яё]", label) and state.value not in label


def test_board_has_one_blank_line_before_each_task_and_none_elsewhere() -> None:
    keys = replace(TASK, id=2, title="Ключи")
    waiting = talking(last_out=T0, last_in=None)
    card = board_card([TASK, keys], {1: [(waiting, CONTACT)], 2: []}, T0)
    assert card.text == (
        "📋 Статус задач\n"
        "\n#1 3D-печать — в работе\n"
        "• @print_lab — жду ответа с 10:00\n"
        "\n#2 Ключи — в работе\n"
        "• людей пока нет"
    )


QUESTION = {
    "text": "Какой корпус нужен?",
    "last_incoming": "Сколько?",
    "task": "Цена печати 5 корпусов из PETG",
    "about": "мастерская 3D-печати",
    "received": ["А вам какой нужен?", "Сколько?"],
    "suggested": ["Корпус для платы 10×6×3 см, файл пришлю"],
}


def test_question_card_is_in_said_format() -> None:
    # Саид, 2026-11-09: «Вопрос» in bold and the task, who asks and who they are, what came
    # in, the suggested reply and one «Ок»; any other answer is a reply to the card.
    card = approval_card(approval(ApprovalKind.QUESTION, **QUESTION), CONV, CONTACT)
    assert card.text == (
        "❓ Вопрос — Цена печати 5 корпусов из PETG\n"
        "👤 @print_lab — мастерская 3D-печати\n"
        "✉️ Получено\n"
        "• «А вам какой нужен?»\n"
        "• «Сколько?»\n"
        "💬 Предлагаемый ответ\n"
        "• «Корпус для платы 10×6×3 см, файл пришлю»"
    )
    assert bolded(card) == ["Вопрос"]
    assert card.buttons == [[("Ок", f"ap:{BIG}:ok:abcdef")]]


def test_question_card_without_a_suggestion_has_dashes_and_no_button() -> None:
    payload = {**QUESTION, "about": "", "received": [], "suggested": []}
    card = approval_card(approval(ApprovalKind.QUESTION, **payload), CONV, CONTACT)
    assert card.text == (
        "❓ Вопрос — Цена печати 5 корпусов из PETG\n"
        "👤 @print_lab\n"
        "✉️ Получено\n"
        "—\n"
        "💬 Предлагаемый ответ\n"
        "—"
    )
    assert card.buttons == []


def test_cards_say_what_they_are() -> None:
    # The chat log (control/chatlog.py) records each card's kind; the board is never logged.
    draft = next(all_cards())
    assert draft.kind == "task" and draft.approval_id is None
    assert board_card([TASK], {1: []}, T0).kind == "board"
    assert board_card([], {}, T0).kind == "board"
    assert Card("x").kind == "" and Card("x").approval_id is None


@pytest.mark.parametrize("kind", CORE_KINDS)
def test_an_approval_card_carries_its_approval(kind: ApprovalKind) -> None:
    card = approval_card(approval(kind, text="?"), CONV, CONTACT)
    assert card.kind == "approval" and card.approval_id == BIG


def test_a_question_card_carries_its_approval_too() -> None:
    card = approval_card(approval(ApprovalKind.QUESTION, **QUESTION), CONV, CONTACT)
    assert card.kind == "approval" and card.approval_id == BIG


# The agent core's question card (stage 2): the agent's own question is always shown.
CORE_QUESTION = {
    "task": "Дрель",
    "about": "",
    "question": "Брать дрель за 450 000?",
    "received": [],
    "suggested": ["Беру, когда доставите?"],
}


def test_a_question_card_without_received_messages_shows_the_question() -> None:
    card = approval_card(
        approval(ApprovalKind.QUESTION, **CORE_QUESTION), CONV, CONTACT
    )
    assert card.text == (
        "❓ Вопрос — Дрель\n"
        "👤 @print_lab\n"
        "❓ Брать дрель за 450 000?\n"
        "💬 Предлагаемый ответ\n"
        "• «Беру, когда доставите?»"
    )
    assert card.buttons == [[("Ок", f"ap:{BIG}:ok:abcdef")]]
    assert card.kind == "approval" and card.approval_id == BIG


def test_a_question_card_with_received_messages_shows_the_question_too() -> None:
    payload = {**CORE_QUESTION, "received": ["Есть, 450 000", "Берёте?"]}
    card = approval_card(approval(ApprovalKind.QUESTION, **payload), CONV, CONTACT)
    assert card.text == (
        "❓ Вопрос — Дрель\n"
        "👤 @print_lab\n"
        "❓ Брать дрель за 450 000?\n"
        "✉️ Получено\n"
        "• «Есть, 450 000»\n"
        "• «Берёте?»\n"
        "💬 Предлагаемый ответ\n"
        "• «Беру, когда доставите?»"
    )


def test_a_question_about_the_task_as_a_whole_has_no_person_line() -> None:
    payload = {**CORE_QUESTION, "suggested": []}
    card = approval_card(approval(ApprovalKind.QUESTION, **payload), None, None)
    assert card.text == (
        "❓ Вопрос — Дрель\n❓ Брать дрель за 450 000?\n💬 Предлагаемый ответ\n—"
    )
    assert card.buttons == []


# The agent core's order card: what «Да» approves is on the card.
CORE_ORDER = {
    "text": "Заказать дрель?",
    "amount": 240_000,
    "terms": "доставка завтра",
    "currency": "UZS",
    "last_incoming": "Берёте?",
    "buttons": [[["Да", "yes"], ["Нет", "no"]], [["Напишу сам", "self"]]],
}


def test_an_order_card_shows_the_amount_and_the_terms() -> None:
    card = approval_card(
        approval(ApprovalKind.CONFIRM_ORDER, **CORE_ORDER), CONV, CONTACT
    )
    assert card.text.splitlines()[:4] == [
        "❓ Подтвердить заказ? · @print_lab",
        "Заказать дрель?",
        "Сумма: 240 000 сум",
        "Условия: доставка завтра",
    ]
    assert card.buttons == [
        [
            ("Да — 240 000 сум", f"ap:{BIG}:yes:abcdef"),
            ("Нет", f"ap:{BIG}:no:abcdef"),
        ],
        [("Напишу сам", f"ap:{BIG}:self:abcdef")],
    ]


def test_an_order_card_with_terms_only() -> None:
    payload = {**CORE_ORDER, "amount": None}
    card = approval_card(approval(ApprovalKind.CONFIRM_ORDER, **payload), CONV, CONTACT)
    assert "Сумма" not in card.text
    assert "Условия: доставка завтра" in card.text
    assert card.buttons[0][0] == ("Да", f"ap:{BIG}:yes:abcdef")


def test_an_order_card_without_buttons_in_its_payload_gets_yes_no_and_mine() -> None:
    card = approval_card(
        approval(ApprovalKind.CONFIRM_ORDER, text="Заказать?", last_incoming="Да"),
        CONV,
        CONTACT,
    )
    assert card.text.splitlines() == [
        "❓ Подтвердить заказ? · @print_lab",
        "Заказать?",
        "От собеседника: «Да»",
        "Можно ответить на эту карточку своими словами — агент получит их как указание.",
    ]
    assert card.buttons == [
        [("Да", f"ap:{BIG}:yes:abcdef"), ("Нет", f"ap:{BIG}:no:abcdef")],
        [("Напишу сам", f"ap:{BIG}:self:abcdef")],
    ]


# --- stage 3: the agent's line, each person's note and the agent core's states --------------

NOW = T0 + timedelta(hours=4, minutes=30)  # 14:30 in Tashkent
CORE_TASK = replace(
    TASK,
    id=7,
    title="Печать корпусов",
    headline="Ответили двое из трёх: PrintLab — 240 000, 3 дня; Maker — 310 000, 2 дня",
)


def person(conv_id: int, username: str | None, name: str = "Иван") -> Contact:
    return replace(
        CONTACT, id=conv_id, peer_id=2000 + conv_id, username=username,
        display_name=name,
    )  # fmt: skip


def core_conv(conv_id: int, state: ConvState, **changes: object) -> Conversation:
    fields: dict[str, object] = {
        "facts": {}, "last_in_at": None, "last_out_at": None, **changes,
    }  # fmt: skip
    return replace(CONV, id=conv_id, task_id=7, state=state, **fields)


def core_people() -> list[tuple[Conversation, Contact | None]]:
    at = NOW.replace(hour=13, minute=40)
    return [
        (core_conv(1, ConvState.PLANNED, first_sent=False), person(1, "new_one")),
        (
            core_conv(
                2,
                ConvState.TALKING,
                last_out_at=at,
                last_in_at=NOW.replace(minute=15),
                note="240 000, 3 дня",
            ),
            person(2, "print_lab"),
        ),
        (core_conv(3, ConvState.CONTACTED, last_out_at=at), person(3, "maker_uz")),
        (core_conv(4, ConvState.WAITING_OWNER, last_out_at=at), person(4, None)),
        (
            core_conv(5, ConvState.CLOSED, close_reason="отказался", last_out_at=at),
            person(5, "old"),
        ),
    ]


def test_the_board_shows_the_headline_notes_and_the_core_states() -> None:
    card = board_card([CORE_TASK], {7: core_people()}, NOW)
    assert card.text.splitlines() == [
        "📋 Статус задач",
        "",
        "#7 Печать корпусов — в работе",
        "Ответили двое из трёх: PrintLab — 240 000, 3 дня; Maker — 310 000, 2 дня",
        "• @new_one — ещё не написали",
        "• @print_lab — ответил в 14:15 · 240 000, 3 дня",
        "• @maker_uz — жду ответа с 13:40",
        "• Иван — ждёт твоего решения",
        "• @old — закончили: отказался",
    ]
    assert bolded(card) == ["Статус задач", "#7 Печать корпусов — в работе"]
    assert card.buttons == [[("Закрыть #7 Печать корпусов", "fn:7:ask:-")]]


def test_a_paused_task_says_so() -> None:
    paused = replace(CORE_TASK, status=TaskStatus.PAUSED)
    card = board_card([paused], {7: []}, NOW)
    assert card.text.splitlines()[2] == "#7 Печать корпусов — на паузе"


def test_a_board_without_headline_and_notes_reads_as_before() -> None:
    # The old engine's data, as in test_board_tells_waiting_for_a_reply_from_a_reply_received.
    waiting = talking(last_out=T0, last_in=None)
    answered = talking(
        last_out=T0, last_in=T0 + timedelta(minutes=5), facts={"price": "240 000 сум"}
    )
    text = board_card(
        [TASK], {1: [(waiting, CONTACT), (answered, OTHER)]}, T0 + timedelta(hours=1)
    ).text
    assert text == (
        "📋 Статус задач\n"
        "\n#1 3D-печать — в работе\n"
        "• @print_lab — жду ответа с 10:00\n"
        "• @3d_tash — ответил в 10:05"
    )


def test_the_old_engines_facts_are_not_shown_only_the_note() -> None:
    conv = replace(
        talking(last_out=T0, last_in=None, facts={"price": "240 000 сум"}),
        note="торгуется",
    )
    text = board_card([TASK], {1: [(conv, CONTACT)]}, T0).text
    assert text.splitlines()[-1] == "• @print_lab — жду ответа с 10:00 · торгуется"


def test_a_long_note_and_a_long_headline_are_cut() -> None:
    task = replace(CORE_TASK, headline="ш" * 300)
    conv = core_conv(2, ConvState.WAITING_OWNER, note="н" * 300)
    lines = board_card([task], {7: [(conv, person(2, "p"))]}, NOW).text.splitlines()
    assert lines[3] == "ш" * 199 + "…"
    assert lines[4] == "• @p — ждёт твоего решения · " + "н" * 149 + "…"
    short = replace(conv, note="н" * 150)
    line = board_card([task], {7: [(short, person(2, "p"))]}, NOW).text.splitlines()[4]
    assert line.endswith(" · " + "н" * 150)


def test_conv_stage_of_a_closed_conversation() -> None:
    closed = replace(CONV, state=ConvState.CLOSED, close_reason=None)
    assert conv_stage(closed) == "закончили"
    assert conv_stage(replace(closed, close_reason="")) == "закончили"
    assert conv_stage(replace(closed, close_reason="занят")) == "закончили: занят"
    assert conv_stage(replace(CONV, state=ConvState.PLANNED)) == "ещё не написали"


def test_a_first_message_that_waits_says_when_it_goes() -> None:
    queued = core_conv(3, ConvState.CONTACTED, first_sent=False)
    sent = core_conv(4, ConvState.CONTACTED, first_sent=True, last_out_at=NOW)
    people = [(queued, person(3, "maker_uz")), (sent, person(4, "sent"))]
    zone = NOW.tzinfo

    def lines(goes: datetime) -> list[str]:
        mapping = {3: goes, 4: goes}
        text = board_card([CORE_TASK], {7: people}, NOW, goes_at=mapping).text
        return text.splitlines()[4:]

    today = NOW.replace(hour=14, minute=50).astimezone(ZoneInfo("UTC"))
    assert lines(today) == [
        "• @maker_uz — напишу в 14:50",
        "• @sent — жду ответа с 14:30",
    ]
    tomorrow = (NOW + timedelta(days=1)).replace(hour=9, minute=5, tzinfo=zone)
    assert lines(tomorrow)[0] == "• @maker_uz — напишу завтра в 09:05"
    later = (NOW + timedelta(days=2)).replace(hour=9, minute=5, tzinfo=zone)
    assert lines(later)[0] == "• @maker_uz — напишу 12.11 в 09:05"
    # Without the mapping, or for a person who is not in it, the line is as before.
    plain = board_card([CORE_TASK], {7: people}, NOW).text.splitlines()[4]
    assert plain == "• @maker_uz — жду ответа"
    other = board_card([CORE_TASK], {7: people}, NOW, goes_at={99: today})
    assert other.text.splitlines()[4] == "• @maker_uz — жду ответа"


def many_people(task_id: int, count: int, *, closed: int) -> list:
    rows: list[tuple[Conversation, Contact | None]] = []
    for n in range(count):
        state = ConvState.CLOSED if n < closed else ConvState.WAITING_OWNER
        conv = replace(
            core_conv(task_id * 100 + n, state, close_reason="отказался"),
            task_id=task_id, note="з" * 140,
        )  # fmt: skip
        rows.append((conv, person(task_id * 100 + n, f"user{task_id}_{n}")))
    return rows


def test_a_long_board_collapses_closed_people_and_shrinks_notes() -> None:
    tasks = [replace(CORE_TASK, id=n, title=f"Задача {n}") for n in (1, 2, 3)]
    convs = {n: many_people(n, 10, closed=4) for n in (1, 2, 3)}
    card = board_card(tasks, convs, NOW)
    assert len(card.text) <= 3800
    lines = card.text.splitlines()
    assert "• закончили с 4" in lines
    assert not any("закончили: отказался" in line for line in lines)
    assert "• @user1_4 — ждёт твоего решения · " + "з" * 59 + "…" in lines
    # The collapsed line is the last line of each task.
    assert lines[lines.index("#2 Задача 2 — в работе") - 2] == "• закончили с 4"
    assert lines[-1] == "• закончили с 4"
    assert [row[0][1] for row in card.buttons] == [f"fn:{n}:ask:-" for n in (1, 2, 3)]


def shown_tasks(card: Card, ids: range) -> list[int]:
    lines = card.text.splitlines()
    return [n for n in ids if f"#{n} Задача {n} — в работе" in lines]


def test_a_board_still_too_long_names_the_tasks_that_did_not_fit() -> None:
    tasks = [replace(CORE_TASK, id=n, title=f"Задача {n}") for n in range(1, 21)]
    convs = {n: many_people(n, 4, closed=0) for n in range(1, 21)}
    card = board_card(tasks, convs, NOW)
    assert len(card.text) <= 3800
    lines = card.text.splitlines()
    shown = shown_tasks(card, range(1, 21))
    assert shown == list(range(1, len(shown) + 1)) and 1 < len(shown) < 20
    skipped = ", ".join(f"#{n}" for n in range(len(shown) + 1, 21))
    first = len(shown) + 1
    assert lines[-1] == (
        f"Не поместились: {skipped}. Спроси меня, например: «что по задаче {first}?»"
    )
    assert lines[-2] == ""
    assert [row[0][1] for row in card.buttons] == [f"fn:{n}:ask:-" for n in shown]


def test_one_big_task_is_cut_and_does_not_hide_the_others() -> None:
    big = replace(CORE_TASK, id=1, title="Задача 1")
    small = replace(CORE_TASK, id=2, title="Задача 2")
    convs = {1: many_people(1, 40, closed=0), 2: many_people(2, 1, closed=0)}
    card = board_card([big, small], convs, NOW)
    assert len(card.text) <= 4096
    lines = card.text.splitlines()
    assert shown_tasks(card, range(1, 3)) == [1, 2]
    second = lines.index("#2 Задача 2 — в работе")
    block = "\n".join(lines[2 : second - 1])
    assert len(block) <= 1200
    people = [line for line in lines[2 : second - 1] if line.startswith("• @user1_")]
    assert lines[second - 2] == f"• …и ещё {40 - len(people)} человек: спроси «что по задаче 1?»"
    assert 0 < len(people) < 40
    assert lines[-1].startswith("• @user2_0")  # nothing skipped: no «Не поместились»
    assert [row[0][1] for row in card.buttons] == ["fn:1:ask:-", "fn:2:ask:-"]


def test_a_later_smaller_task_is_shown_after_one_that_did_not_fit() -> None:
    ids = range(1, 6)
    tasks = [replace(CORE_TASK, id=n, title=f"Задача {n}") for n in ids]
    convs = {n: many_people(n, 10, closed=0) for n in range(1, 5)}
    convs[5] = many_people(5, 1, closed=0)
    card = board_card(tasks, convs, NOW)
    assert len(card.text) <= 3800
    assert shown_tasks(card, ids) == [1, 2, 3, 5]
    assert card.text.splitlines()[-1] == "Не поместились: #4. Спроси меня, например: «что по задаче 4?»"
    assert [row[0][1] for row in card.buttons] == [
        f"fn:{n}:ask:-" for n in (1, 2, 3, 5)
    ]


def test_the_agents_text_cannot_forge_a_line() -> None:
    # The headline, a note and a close reason are the agent's words, written after it read
    # counterparts' messages: a line break in them must not start a line of its own.
    task = replace(CORE_TASK, headline="Итог:\n• @fake — ответил в 10:00")
    noted = core_conv(2, ConvState.WAITING_OWNER, note="a\n• @fake — ответил в 10:00")
    closed = core_conv(
        3, ConvState.CLOSED, close_reason="отказался\r\n\t• @fake — ответил"
    )
    people = [(noted, person(2, "p")), (closed, person(3, "q"))]
    card = board_card([task], {7: people}, NOW)
    assert card.text.splitlines()[3:] == [
        "Итог: • @fake — ответил в 10:00",
        "• @p — ждёт твоего решения · a • @fake — ответил в 10:00",
        "• @q — закончили: отказался • @fake — ответил",
    ]


def test_a_first_message_already_late_goes_soon() -> None:
    queued = core_conv(3, ConvState.CONTACTED, first_sent=False)
    late = NOW - timedelta(minutes=10)
    card = board_card(
        [CORE_TASK], {7: [(queued, person(3, "maker_uz"))]}, NOW, goes_at={3: late}
    )
    assert card.text.splitlines()[4] == "• @maker_uz — напишу в ближайшее время"


# -- the stopped agent on the board (stage 3, fix D2.1) --

STOPPED_LINE = "⏸ Агент остановлен — ничего не пишет и не отвечает"
RESUME_ROW = [("Продолжить работу", "rs:0:go:-")]


def test_a_stopped_board_says_so_first_and_offers_to_resume() -> None:
    plain = board_card([TASK], {1: []}, T0)
    card = board_card([TASK], {1: []}, T0, stopped=True)
    title, rest = plain.text.split("\n", 1)
    assert card.text == f"{title}\n{STOPPED_LINE}\n{rest}"
    assert card.buttons == [RESUME_ROW, *plain.buttons]
    # The bold headings move with the text.
    assert bolded(card) == ["Статус задач", "#1 3D-печать — в работе"]
    assert card.kind == "board"


def test_a_stopped_board_without_tasks_still_says_so() -> None:
    card = board_card([], {}, T0, stopped=True)
    assert card.text == f"📋 Задач в работе нет\n{STOPPED_LINE}"
    assert card.buttons == [RESUME_ROW]
    assert card.kind == "board"


def test_a_board_that_is_not_stopped_is_as_before() -> None:
    assert board_card([TASK], {1: []}, T0, stopped=False) == board_card(
        [TASK], {1: []}, T0
    )
    assert board_card([], {}, T0, stopped=False).text == "📋 Задач в работе нет"
    assert board_card([], {}, T0, stopped=False).buttons == []


def test_a_board_stopped_by_telegram_says_why_and_warns_before_the_button() -> None:
    # One tap on «Продолжить работу» would send again into a restricted account.
    card = board_card([], {}, T0, stopped=True, stopped_why="peer_flood")
    assert card.text == (
        "📋 Задач в работе нет\n"
        "⏸ Агент остановлен: Telegram ограничил аккаунт (peer_flood)."
        " Продолжай, только когда ограничение снято"
    )
    assert card.buttons == [RESUME_ROW]
    # the owner's own /stop keeps its line
    lines = board_card([], {}, T0, stopped=True, stopped_why="").text.split("\n")
    assert lines[1] == STOPPED_LINE


def test_a_person_added_by_phone_shows_his_name_cut_and_his_number() -> None:
    # Fix pass E1.6, M10.
    by_phone = replace(
        CONTACT,
        username=None,
        display_name="Алишер\nЛучший мастер 3D-печати во всём Ташкенте и области",
        card={"contact": "+998901234567"},
    )
    card = board_card([TASK], {1: [(talking(last_out=T0, last_in=None), by_phone)]}, T0)
    [line] = [line for line in card.text.split("\n") if line.startswith("• ")]
    assert line.startswith(
        "• Алишер Лучший мастер 3D-печати во всём… (+998 90 123 45 67) — "
    )


# -- stage 5: settled cards (R55) and markers (R54) --


def settled(kind: ApprovalKind, status: ApprovalStatus, answer=None, **payload):
    row = replace(approval(kind, **payload), status=status, answer=answer)
    return settled_approval_card(row, CONTACT)


def test_a_settled_question_says_what_was_asked_and_what_he_answered() -> None:
    long = "Брать " + "очень " * 40 + "дорого?"
    card = settled(
        ApprovalKind.QUESTION, ApprovalStatus.ANSWERED, "Бери",
        task="Дрель", question=long,
    )  # fmt: skip
    first, second = card.text.split("\n")
    assert first.startswith("Вопрос — Дрель: «Брать очень") and first.endswith("…»")
    assert len(first) <= len("Вопрос — Дрель: «»") + 120
    assert second == "Ответ: «Бери»" and card.buttons == []
    mine = settled(
        ApprovalKind.QUESTION, ApprovalStatus.ANSWERED, None, task="Дрель", question="?"
    )
    assert mine.text == "Вопрос — Дрель: «?»\nОтвет: чат у тебя"


def test_settled_orders_phones_addresses_and_people() -> None:
    order = {"text": "Заказать?", "amount": 240_000, "currency": "UZS"}
    yes = settled(ApprovalKind.CONFIRM_ORDER, ApprovalStatus.APPROVED, **order)
    assert yes.text == "Заказ · @print_lab: 240 000 сум — подтверждён"
    no = settled(ApprovalKind.CONFIRM_ORDER, ApprovalStatus.REJECTED, **order)
    assert no.text == "Заказ · @print_lab: 240 000 сум — отклонён"
    words = settled(
        ApprovalKind.CONFIRM_ORDER, ApprovalStatus.ANSWERED, "дешевле", **order
    )
    assert words.text == "Заказ · @print_lab: 240 000 сум — ответил словами"
    phone = settled(ApprovalKind.SHARE_PHONE, ApprovalStatus.APPROVED)
    assert phone.text == "Телефон · @print_lab: дали"
    address = settled(ApprovalKind.SHARE_ADDRESS, ApprovalStatus.REJECTED)
    assert address.text == "Адрес · @print_lab: не дали"


def test_an_expired_card_keeps_its_title_and_was_not_needed() -> None:
    phone = settled(ApprovalKind.SHARE_PHONE, ApprovalStatus.EXPIRED)
    assert phone.text == "Дать твой телефон? · @print_lab — не понадобилось"
    question = settled(
        ApprovalKind.QUESTION, ApprovalStatus.EXPIRED, task="Дрель", question="?"
    )
    assert question.text == "Вопрос — Дрель — не понадобилось"


def test_waiting_cards_carry_the_marker_and_lose_it_when_settled() -> None:
    for kind in (ApprovalKind.SHARE_PHONE, ApprovalKind.CONFIRM_ORDER):
        card = approval_card(approval(kind, text="?"), CONV, CONTACT)
        assert card.text[0] == "❓"
        assert unmarked(card).text[0] != "❓" and unmarked(card).buttons == []
    hand_back = approval_card(
        approval(ApprovalKind.HANDOFF, text="Чат у тебя"), CONV, CONTACT
    )
    assert hand_back.text.startswith("Напиши сам")
    question = approval_card(
        approval(ApprovalKind.QUESTION, task="Дрель", question="?", suggested=[]),
        CONV,
        CONTACT,
    )
    assert bolded(question) == ["Вопрос"]  # the bold range follows the marker
    assert board_card([], {}, T0).text[0] == "📋"
    result = result_card(3, "Дрель", "Купили")
    assert result.text == "✅ Готово — Дрель\nКупили"
    assert bolded(result) == ["Готово — Дрель"]
    assert unmarked(result).text == result.text  # a result keeps «✅»: the task is done


def test_a_card_nobody_needed_keeps_its_first_line() -> None:
    card = Card("❓ Закрыть задачу #3 «Дрель»?\nАгент её уже закончил", [[("Да", "x")]])
    assert not_needed(card).text == "Закрыть задачу #3 «Дрель»? — не понадобилось"
    launched = launched_card(3, "Дрель", ["Кому писать — ищу сам."])
    assert launched.text == "Задача #3 «Дрель» запущена\nКому писать — ищу сам."
    assert launched.buttons == [[("Закрыть задачу", "fn:3:ask:-")]]
