"""The cases of the secretary's evals (`evals/secretary/cases.py`, `evals/secretary_cases/`):
the format, its mistakes, and the set that ships."""

from __future__ import annotations

import pytest

from tg_agent.evals.secretary.cases import (
    CaseError,
    load_cases,
    parse_case,
)

MINIMAL = """
id = "thanks"
about = "«спасибо» — ничего не делать"
messages = ["спасибо"]

[expect]
drafts_made = 0
"""

FULL = """
id = "full"
about = "всё сразу"
stopped = true
messages = ["по задаче 3 спроси про доставку", "и цену тоже"]
reply_to_draft = 1

[[drafts]]
id = 1
title = "Футболки"
brief = "Напечатать 50 футболок"
people = ["@print_lab"]
budget = 3000000
missing = ["размеры"]
words = "надо 50 футболок, @print_lab"

[[tasks]]
id = 3
title = "Упаковка"
brief = "Найти поставщика коробок"
status = "paused"
headline = "Жду ответа от двух поставщиков"

[[tasks.people]]
username = "pack_uz"
state = "waiting_owner"
messages = [
  { from = "agent", text = "Здравствуйте! Сколько стоят 500 коробок?" },
  { from = "counterpart", text = "Белые или крафт?" },
]

[[finished]]
id = 5
title = "Торт"
brief = "Заказать торт на субботу"
headline = "Готово: торт заказан, 450 000 сум"

[[cards]]
id = 1
task = 3
kind = "question"
about = "pack_uz"
question = "Коробки белые или крафт?"

[[cards]]
id = 2
task = 3
kind = "question"
question = "Искать ещё поставщиков?"

[[chat]]
from = "bot"
text = "📋 Задачи: …"

[[chat]]
from = "owner"
text = "ок"

[expect]
only_tools = ["instruct_task"]
quotes_ok = true

[[expect.calls]]
tool = "instruct_task"
args = { task_id = 3, quote = { has = "доставк" } }
"""


def test_a_minimal_case():
    case = parse_case(MINIMAL)
    assert case.id == "thanks"
    assert case.messages == ("спасибо",)
    assert case.drafts == case.tasks == case.cards == case.chat == ()
    assert case.stopped is False
    assert case.reply_to_draft is None
    assert case.expect.drafts_made == 0


def test_a_full_case():
    case = parse_case(FULL)
    assert case.stopped is True
    assert case.reply_to_draft == 1
    (draft,) = case.drafts
    assert (draft.id, draft.people, draft.budget, draft.missing) == (
        1, ("@print_lab",), 3000000.0, ("размеры",),
    )  # fmt: skip
    assert draft.words == "надо 50 футболок, @print_lab"
    paused, done = case.tasks
    assert (paused.status, done.status) == ("paused", "done")
    (person,) = paused.people
    assert person.username == "pack_uz"
    assert person.state == "waiting_owner"
    assert person.messages == (
        ("agent", "Здравствуйте! Сколько стоят 500 коробок?"),
        ("counterpart", "Белые или крафт?"),
    )
    assert [card.kind for card in case.cards] == ["question", "question"]
    assert [card.about for card in case.cards] == ["pack_uz", None]
    assert [row.speaker for row in case.chat] == ["bot", "owner"]
    assert case.expect.only_tools == ("instruct_task",)


def test_a_draft_without_words_was_written_as_its_brief():
    case = parse_case(FULL.replace('words = "надо 50 футболок, @print_lab"\n', ""))
    assert case.drafts[0].words == "Напечатать 50 футболок"


def bad(text: str, match: str) -> None:
    with pytest.raises(CaseError, match=match):
        parse_case(text, "x.toml")


def test_mistakes_name_the_file_and_the_key():
    bad("id = 1", "x.toml")
    bad(MINIMAL + "\nsurprise = 1\n", "surprise")
    bad(MINIMAL.replace('["спасибо"]', "[]"), "messages")
    bad(MINIMAL.replace('["спасибо"]', '["  "]'), "messages")
    bad(MINIMAL.replace("drafts_made = 0", ""), "expect")
    bad(MINIMAL.replace("drafts_made", "drafts_maked"), "drafts_maked")
    bad("это не toml", "TOML")


def test_task_numbers_are_unique_and_positive():
    bad(FULL.replace("id = 5", "id = 3"), "3")
    bad(FULL.replace("id = 5", "id = 0"), "id")


def test_reply_to_draft_names_a_draft():
    bad(FULL.replace("reply_to_draft = 1", "reply_to_draft = 3"), "reply_to_draft")


def test_cards_are_numbered_from_one_in_order():
    bad(FULL.replace("id = 2\ntask = 3", "id = 3\ntask = 3"), "cards")


def test_a_card_is_about_a_task_in_work_or_finished_and_one_of_its_people():
    bad(
        FULL.replace('task = 3\nkind = "question"', 'task = 1\nkind = "question"'),
        "task",
    )
    bad(
        FULL.replace('about = "pack_uz"\nquestion', 'about = "nobody_here"\nquestion'),
        "nobody_here",
    )
    bad(FULL.replace('kind = "question"', 'kind = "riddle"'), "kind")


def test_a_person_has_a_username_a_known_state_and_known_speakers():
    bad(FULL.replace('state = "waiting_owner"', 'state = "asleep"'), "state")
    bad(FULL.replace('username = "pack_uz"', 'username = "@p"'), "username")
    bad(FULL.replace('from = "agent"', 'from = "owner"'), "from")


def test_a_task_status_is_active_or_paused():
    bad(FULL.replace('status = "paused"', 'status = "done"'), "status")


def test_chat_rows_are_the_owner_s_or_the_bot_s():
    bad(FULL.replace('from = "bot"', 'from = "agent"'), "chat")


# -- the set that ships --

CHECK_LIST = (
    "status_question", "thanks", "two_tasks_one_message", "draft_correction",
    "launch_anyway", "instruct_running", "answers_only_question",
    "new_task_while_question_open",
    "burst_three_messages_one_task", "burst_request_then_cancel",
    "burst_launch_then_stop", "burst_city_correction",
    "close_task_in_words", "pause_task_in_words", "resume_when_stopped",
    "what_did_username_answer",
    "small_talk_no_draft", "counterpart_quote_launch_all", "headline_reads_like_order",
    "owner_asks_budget", "excluded_contact", "budget_in_words",
)  # fmt: skip


def test_the_set_has_thirty_cases_or_more_each_named_as_its_file():
    cases = load_cases()
    assert len(cases) >= 30
    assert len({case.id for case in cases}) == len(cases)


def test_the_set_covers_the_brief():
    ids = {case.id for case in load_cases()}
    assert set(CHECK_LIST) <= ids


def test_the_set_has_bursts_and_uzbek_in_latin_letters():
    cases = load_cases()
    assert sum(len(case.messages) > 1 for case in cases) >= 4
    uzbek = [case for case in cases if case.id.startswith("uz_")]
    assert len(uzbek) >= 3


def test_the_owner_writes_short_messages():
    for case in load_cases():
        for message in case.messages:
            assert len(message) <= 200, (case.id, message)


def test_only_loads_the_named_cases_in_order_and_refuses_an_unknown_one():
    assert [case.id for case in load_cases(["thanks", "status_question"])] == [
        "thanks", "status_question",
    ]  # fmt: skip
    with pytest.raises(CaseError, match="nope"):
        load_cases(["nope"])
