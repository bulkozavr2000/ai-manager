"""The setup commands: secrets into the keyring, the Telegram login and the check.

Nothing here reaches the network: the clients are stubs. No secret may ever be printed.
"""

from __future__ import annotations

from types import SimpleNamespace

from tg_agent.config import Settings
from tg_agent.secrets import Key, get_secret, set_secret
from tg_agent.setup.__main__ import ask_secrets, check, login

ALL = {
    Key.ANTHROPIC_API_KEY: "sk-ant-secret",
    Key.BOT_TOKEN: "7123456789:bot-secret",
    Key.TG_API_ID: "12345",
    Key.TG_API_HASH: "hash-secret",
    Key.TG_SESSION: "session-secret",
}


class StubTelegram:
    """Just enough of a Telethon client for the setup commands."""

    def __init__(self, *, authorized: bool = True) -> None:
        self.authorized = authorized
        self.started = self.disconnected = False
        self.session = SimpleNamespace(save=lambda: "new-session-secret")

    async def start(self) -> None:
        self.started = True

    async def connect(self) -> None:
        pass

    async def is_user_authorized(self) -> bool:
        return self.authorized

    async def get_me(self) -> SimpleNamespace:
        return SimpleNamespace(
            first_name="Саид",
            last_name=None,
            username="said_work",
            phone="998901234567",
        )

    async def disconnect(self) -> None:
        self.disconnected = True


class StubAnthropic:
    def __init__(self, error: Exception | None = None) -> None:
        async def retrieve(model: str) -> SimpleNamespace:
            if error is not None:
                raise error
            return SimpleNamespace(id=model)

        self.models = SimpleNamespace(retrieve=retrieve)


class StubBot:
    async def get_me(self) -> SimpleNamespace:
        return SimpleNamespace(username="said_agent_bot")


def no_secret_in(lines: list[str]) -> bool:
    text = "\n".join(lines)
    return not any(value in text for value in [*ALL.values(), "new-session-secret"])


# --- secrets ---------------------------------------------------------------------------------


def test_secrets_asks_only_for_what_is_missing() -> None:
    set_secret(Key.BOT_TOKEN, ALL[Key.BOT_TOKEN])
    answers = iter(["sk-ant-secret", "12345", "hash-secret", "", ""])
    asked: list[str] = []
    said: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return next(answers)

    saved = ask_secrets(force=False, ask=ask, say=said.append)

    assert saved == [Key.ANTHROPIC_API_KEY, Key.TG_API_ID, Key.TG_API_HASH]
    assert len(asked) == 5  # the bot token was already there
    assert get_secret(Key.BOT_TOKEN) == ALL[Key.BOT_TOKEN]
    assert get_secret(Key.OWNER_PHONE) is None  # optional, left empty
    assert no_secret_in(said)


def test_secrets_force_replaces_and_insists_on_a_numeric_api_id() -> None:
    set_secret(Key.TG_API_ID, "1")
    answers = iter(["k", "t", "abc", "777", "h", "+998 90 000 00 00", "Ташкент"])

    saved = ask_secrets(force=True, ask=lambda _: next(answers), say=lambda _: None)

    assert get_secret(Key.TG_API_ID) == "777"
    assert len(saved) == 6


# --- login -----------------------------------------------------------------------------------


async def test_login_keeps_the_session_and_shows_only_the_name() -> None:
    set_secret(Key.TG_API_ID, "12345")
    set_secret(Key.TG_API_HASH, "hash-secret")
    client, said = StubTelegram(), []

    await login(force=False, client=client, say=said.append)

    assert get_secret(Key.TG_SESSION) == "new-session-secret"
    assert client.started and client.disconnected
    assert "Саид" in said[-1] and "998901234567" not in "\n".join(said)
    assert no_secret_in(said)


async def test_login_keeps_an_existing_session_without_force() -> None:
    for key, value in ALL.items():
        set_secret(key, value)
    client = StubTelegram()

    await login(force=False, client=client, say=lambda _: None)

    assert not client.started
    assert get_secret(Key.TG_SESSION) == ALL[Key.TG_SESSION]


# --- check -----------------------------------------------------------------------------------


async def test_check_names_missing_secrets_without_showing_the_others(
    settings: Settings,
) -> None:
    set_secret(Key.ANTHROPIC_API_KEY, ALL[Key.ANTHROPIC_API_KEY])
    said: list[str] = []

    assert await check(settings, say=said.append) is False

    assert "bot_token" in "\n".join(said)
    assert no_secret_in(said)


async def test_check_passes_when_every_service_answers(settings: Settings) -> None:
    for key, value in ALL.items():
        set_secret(key, value)
    said: list[str] = []

    ok = await check(
        settings,
        anthropic_client=StubAnthropic(),
        bot=StubBot(),
        telegram=StubTelegram(),
        say=said.append,
    )

    assert ok is True
    assert no_secret_in(said)


async def test_check_fails_on_a_session_telegram_no_longer_accepts(
    settings: Settings,
) -> None:
    for key, value in ALL.items():
        set_secret(key, value)
    said: list[str] = []

    ok = await check(
        settings,
        anthropic_client=StubAnthropic(),
        bot=StubBot(),
        telegram=StubTelegram(authorized=False),
        say=said.append,
    )

    assert ok is False
    assert "setup login" in "\n".join(said)


async def test_check_reports_a_rejected_key_by_its_error_kind_only(
    settings: Settings,
) -> None:
    for key, value in ALL.items():
        set_secret(key, value)
    said: list[str] = []

    ok = await check(
        settings,
        anthropic_client=StubAnthropic(RuntimeError("bad key sk-ant-secret")),
        bot=StubBot(),
        telegram=StubTelegram(),
        say=said.append,
    )

    assert ok is False
    assert "RuntimeError" in "\n".join(said)
    assert no_secret_in(said)
