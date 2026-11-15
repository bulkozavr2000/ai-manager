from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tg_agent.brain.context import (
    counterpart_language,
    escape_untrusted,
    hide_amount,
    load_prompt,
    media_type,
    render_transcript,
)
from tg_agent.models import (
    Contact,
    Conversation,
    ConvState,
    Direction,
    Message,
    MsgSource,
)

TZ = ZoneInfo("Asia/Tashkent")
T0 = datetime(2026, 11, 10, 10, 0, tzinfo=TZ)


def make_conv(**changes: object) -> Conversation:
    fields = {
        "id": 7, "task_id": 1, "contact_id": 3, "state": ConvState.TALKING, "prev_state": None,
        "facts": {"price": "240 000 сум", "price_value": 240000}, "summary": "Спросили цену",
        "language": "ru", "offtopic_strikes": 1, "agent_messages": 2, "followups_sent": 0, "first_sent": True,
        "paused_until": None, "last_in_at": T0, "last_out_at": T0, "created_at": T0, "updated_at": T0,
    }  # fmt: skip
    fields.update(changes)
    return Conversation(**fields)


def make_contact(**changes: object) -> Contact:
    fields = {
        "id": 3, "peer_id": 555, "username": "print_lab", "display_name": "Акмаль | 3D Print", "is_bot": False,
        "language": "ru", "card": {}, "do_not_contact": False, "updated_at": T0,
    }  # fmt: skip
    fields.update(changes)
    return Contact(**fields)


def msg(i: int, text: str, source: MsgSource = MsgSource.COUNTERPART) -> Message:
    direction = Direction.IN if source is MsgSource.COUNTERPART else Direction.OUT
    return Message(
        id=i, conversation_id=7, direction=direction, source=source, tg_msg_id=i, kind="text",
        text=text, created_at=T0 + timedelta(minutes=i), edited_at=None,
    )  # fmt: skip


def test_render_transcript_labels_and_local_time() -> None:
    rendered = render_transcript(
        [msg(1, "Цена?"), msg(2, "240 000 сум", MsgSource.AGENT), msg(3, "ок", MsgSource.OWNER_MANUAL)],
        TZ,
    )  # fmt: skip
    lines = rendered.splitlines()
    assert lines[0] == "[10.11 10:01] Собеседник: Цена?"
    assert lines[1].startswith("[10.11 10:02] Ассистент: ")
    assert lines[2].startswith("[10.11 10:03] Саид (сам): ")


def test_media_messages_are_labelled() -> None:
    voice = msg(1, "")
    voice.kind = "voice"
    rendered = render_transcript([voice], TZ)
    assert "голосовое" in rendered


def test_the_counterparts_language_as_a_code_for_the_gates() -> None:
    unknown = {"conv": make_conv(language=None), "contact": make_contact(language=None)}
    assert (
        counterpart_language(history=[msg(1, "Пришлите контактный номер")], **unknown)
        == "ru"
    )
    assert counterpart_language(history=[], **unknown) is None
    stored_ru = {"conv": make_conv(language="ru"), "contact": make_contact()}
    assert (
        counterpart_language(history=[msg(1, "Нархи қанча?")], **stored_ru) == "uz_cyrl"
    )
    assert counterpart_language(history=[msg(1, "Майли")], **stored_ru) == "ru"


def test_persona_writes_uzbek_in_latin_only() -> None:
    # Саид's rule: Uzbek always in Latin, also to someone who writes in Cyrillic.
    persona = load_prompt("persona")
    assert "«rahmat»" in persona and "латиницей" in persona
    assert "«спасибо», а не «rahmat»" in persona  # no Uzbek words in a Russian reply
    assert "раҳмат" not in persona.casefold() and "Ҳа, мен" not in persona
    assert "не смешивай" in persona


def test_the_model_does_not_introduce_itself() -> None:
    # The account's name says it is an AI («ИИ помощник Саида»); where it does not, the code
    # puts the introduction in (`agent/outgoing.py`), so the model is not asked for one.
    persona = load_prompt("persona")
    assert "представься" not in persona and "«Я ИИ-помощник Саида»" not in persona


def test_asking_said_sends_the_vendor_nothing() -> None:
    # Саид, 2026-11-09: no «Уточню, позже напишу»; ask him, then come back to the vendor.
    assert "Уточню, позже напишу" not in load_prompt("persona")


def test_the_judge_lets_the_agent_say_it_will_write_later() -> None:
    # Without this the judge took «Уточню у Саида и вернусь» for an invented promise, and the
    # chat stalled each time the agent took a question to the owner.
    assert "«уточню, позже напишу»" in load_prompt("judge")


def test_first_message_example_promises_no_file() -> None:
    # The model copies the example, also into tasks that have no file.
    persona = load_prompt("persona")
    [example] = [line for line in persona.splitlines() if "Пример первого" in line]
    assert "файл" not in example.casefold()


def test_prompts_contain_required_rules() -> None:
    persona = load_prompt("persona")
    # Ruling R82: no long template for «вы бот?» any more; the task rules hold one line.
    assert "Да, я ИИ-помощник Саида Ахмедова" not in persona
    assert "Ha, men Said Axmedovning" not in persona
    assert "Собеседник не хочет общаться с ИИ: «Понял, передам Саиду, он напишет сам»" in persona
    rules = load_prompt("task")
    assert (  # ruling R93: and never «нет»
        "2. Спросили, кто ты, не бот ли ты или человек ли ты, — ответь «помощник {owner_gen}»,"
        " без «да» и «нет»"
    ) in rules
    assert "по шаблону из раздела «Шаблоны»" not in rules
    assert "Ассаляму алейкум!" in persona and "Добрый день!" in persona
    assert load_prompt("judge") and "{today}" in load_prompt("secretary")


class _CountingFile:
    """A prompt file that counts how often it is read from the disk."""

    def __init__(self, reads: list[str], name: str) -> None:
        self._reads, self._name = reads, name

    def __truediv__(self, part: str) -> _CountingFile:
        return _CountingFile(self._reads, part)

    def read_text(self, encoding: str) -> str:
        self._reads.append(self._name)
        return f"текст {self._name}"


@pytest.fixture
def counted_reads(monkeypatch: pytest.MonkeyPatch):
    """The prompt files as files that count their reads; the cache is empty before and
    after, so no other test sees these texts."""
    from tg_agent.brain import context

    reads: list[str] = []
    context.load_prompt.cache_clear()
    monkeypatch.setattr(context, "files", lambda package: _CountingFile(reads, package))
    yield reads
    context.load_prompt.cache_clear()


def test_a_prompt_is_read_from_the_disk_once_however_often_it_is_asked_for(
    counted_reads: list[str],
) -> None:
    # Fix pass D1.9: the running bot does not follow edits of the working tree.
    for _ in range(3):
        assert load_prompt("task") == "текст task.md"
    load_prompt("judge")
    assert counted_reads == ["task.md", "judge.md"]


def test_load_prompts_reads_every_prompt_the_program_uses(
    counted_reads: list[str],
) -> None:
    from tg_agent.brain.context import load_prompts

    load_prompts()

    assert sorted(counted_reads) == [
        "judge.md", "persona.md", "search.md", "secretary.md", "task.md",
    ]  # fmt: skip


def test_escape_untrusted() -> None:
    assert escape_untrusted("<a>") == "‹a›"


@pytest.mark.parametrize(
    "goal",
    [
        "Узнать цену печати (бюджет до 300 000 сум)",
        "Узнать цену, бюджет 300000",
        "Узнать цену, не дороже 300 тыс",
        "Узнать цену, максимум 300к",
        "Узнать цену, до 0,3 млн",
    ],
)
def test_the_ceiling_written_into_the_goal_is_hidden(goal: str) -> None:
    # The intake model once put "(бюджет до 300 000 сум)" into the goal, which the model reads.
    text = hide_amount(f"{goal}; корпус, до 300 000 сум", 300_000)
    for leak in ("300 000", "300000", "300 тыс", "300к", "0,3 млн"):
        assert leak not in text


@pytest.mark.parametrize(
    "text", ["5 корпусов", "250 000 сум", "3000 штук", "300 штук по 1 000", "к 5 марта"]
)
def test_hide_amount_keeps_other_numbers(text: str) -> None:
    assert hide_amount(text, 300_000) == text


# Fix pass D1.3: the spellings people really write a budget in.
@pytest.mark.parametrize(
    ("text", "ceiling", "hidden", "other"),
    [
        ("Бюджет 300 тысяч", 300_000, "Бюджет [скрыто]", 400_000),
        ("до 300 тыщ", 300_000, "до [скрыто]", 400_000),
        ("за 300 т.р.", 300_000, "за [скрыто]", 400_000),
        ("за 300 т. р., не больше", 300_000, "за [скрыто], не больше", 400_000),
        ("за 300 тр", 300_000, "за [скрыто]", 400_000),
        ("0.3 миллиона", 300_000, "[скрыто]", 400_000),
        ("1,5 миллиона", 1_500_000, "[скрыто]", 2_500_000),
        ("2 ляма", 2_000_000, "[скрыто]", 3_000_000),
        ("300 000 сум", 300_000, "[скрыто] сум", 400_000),
        ("300'000", 300_000, "[скрыто]", 400_000),
        ("300’000 сум", 300_000, "[скрыто] сум", 400_000),
        ("максимум триста тысяч", 300_000, "максимум [скрыто]", 400_000),
        ("двести пятьдесят тысяч", 250_000, "[скрыто]", 300_000),
        ("полтора миллиона", 1_500_000, "[скрыто]", 2_500_000),
        ("не больше полутора миллионов", 1_500_000, "не больше [скрыто]", 2_000_000),
        ("один миллион двести тысяч", 1_200_000, "[скрыто]", 1_000_000),
        ("у @print_lab, до 300", 300_000, "у @print_lab, до [скрыто]", 400_000),
        ("максимум 2", 2_000_000, "максимум [скрыто]", 3_000_000),
        ("не дороже 300 сум", 300_000, "не дороже [скрыто] сум", 400_000),
        ("не больше 300.", 300_000, "не больше [скрыто].", 400_000),
        ("бюджет у меня 300", 300_000, "бюджет у меня [скрыто]", 400_000),
        ("потолок 300", 300_000, "потолок [скрыто]", 400_000),
        ("в пределах 1,5", 1_500_000, "в пределах [скрыто]", 2_500_000),
        ("уложиться в 300 и не тянуть", 300_000, "уложиться в [скрыто] и не тянуть", 1),
        ("триста тыс. сум", 300_000, "[скрыто]. сум", 400_000),
        ("300 минг сўм", 300_000, "[скрыто] сўм", 400_000),
        ("300 ming so'm", 300_000, "[скрыто] so'm", 400_000),
        ("1,5 million", 1_500_000, "[скрыто]", 2_500_000),
    ],
)
def test_every_spelling_of_the_ceiling_is_hidden_and_only_the_ceiling(
    text: str, ceiling: float, hidden: str, other: float
) -> None:
    assert hide_amount(text, ceiling) == hidden
    assert hide_amount(text, other) == text


@pytest.mark.parametrize(
    "text",
    [
        "5 корпусов размером 300 мм",
        "до 300 мм",  # a bare number with a unit after it is no budget
        "потолок высотой 300 см",
        "Нужно 300 штук до пятницы",
        "до трёхсот",  # not recognised: the trusted block leaves such words out
        "подойдёт 300",  # «до» inside a word is no budget word
        "до 30 сентября, бюджет потом",
    ],
)
def test_a_number_that_is_not_the_ceiling_is_never_hidden(text: str) -> None:
    assert hide_amount(text, 300_000) == text


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (b"\x89PNG\r\n\x1a\n" + bytes(8), "image/png"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"GIF89a" + bytes(8), "image/gif"),
        (b"\xff\xd8\xff\xe0" + bytes(8), "image/jpeg"),
    ],
)
def test_an_image_is_told_by_its_first_bytes(data: bytes, kind: str) -> None:
    assert media_type(data) == kind


@pytest.mark.parametrize(
    ("text", "ceiling"),
    [
        ("позвони до 10:00", 10_000_000),  # a time
        ("скидка до 10%", 10_000),  # a share
        ("срок до 5-7 дней", 5_000),  # a range
        ("до 2/3 суммы вперёд", 2_000_000),  # a fraction
    ],
)
def test_a_bare_number_in_a_time_a_share_or_a_range_is_no_budget(
    text: str, ceiling: float
) -> None:
    assert hide_amount(text, ceiling) == text
