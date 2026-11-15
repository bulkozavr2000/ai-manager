"""Setup from the terminal: the secrets into the keyring, the Telegram login, a check.

    uv run python -m tg_agent.setup secrets [--force]
    uv run python -m tg_agent.setup login [--force]
    uv run python -m tg_agent.setup check
    uv run python -m tg_agent.setup export-keys <path>

Secrets are typed in hidden and kept in the OS keyring; where the system has none (a Linux
server), in `keys.toml` in the data folder, for its owner only. `export-keys` writes that file
from the keyring, to carry the keys to a server. Nothing secret is ever printed: not a key,
not the session, not the account's phone number.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tg_agent.config import PROJECT_ROOT, ConfigError, Settings, load_settings
from tg_agent.main import load_config
from tg_agent.secrets import (
    Key,
    MissingSecret,
    get_secret,
    require_secret,
    set_secret,
    stored_in_key_store,
    use_data_dir,
    write_keys_file,
)

Say = Callable[[str], None]

# What `secrets` asks for, in this order; the session comes from `login`.
PROMPTS = {
    Key.ANTHROPIC_API_KEY: "API-ключ Anthropic (sk-ant-…)",
    Key.BOT_TOKEN: "Токен бота от @BotFather",
    Key.TG_API_ID: "api_id с my.telegram.org (число)",
    Key.TG_API_HASH: "api_hash с my.telegram.org",
    Key.OWNER_PHONE: "Твой телефон: агент даёт его только по кнопке (Enter — пропустить)",
    Key.OWNER_ADDRESS: "Твой адрес доставки: только по кнопке (Enter — пропустить)",
}
OPTIONAL = frozenset({Key.OWNER_PHONE, Key.OWNER_ADDRESS})
REQUIRED = (
    Key.ANTHROPIC_API_KEY,
    Key.BOT_TOKEN,
    Key.TG_API_ID,
    Key.TG_API_HASH,
    Key.TG_SESSION,
)
_OK, _FAIL = (
    "OK  ",
    "ERR ",
)  # ASCII: a redirected Windows console cannot print every sign


def ask_secrets(
    *, force: bool, ask: Callable[[str], str] = getpass.getpass, say: Say = print
) -> list[Key]:
    """Ask for each secret that is not stored yet (each one with `force`); return those saved.

    An optional one is skipped with an empty answer; api_id must be a number.
    """
    saved: list[Key] = []
    place: Path | None = None  # the keys file, where there is no key store
    for key, prompt in PROMPTS.items():
        if get_secret(key) and not force:
            say(f"{key.value}: уже сохранён, пропускаю (--force — заменить)")
            continue
        while True:
            value = ask(f"{prompt}: ").strip()
            if not value and key in OPTIONAL:
                break
            if not value:
                say("Пусто — введи ещё раз")
            elif key is Key.TG_API_ID and not value.isdigit():
                say("api_id — это число, введи ещё раз")
            else:
                place = set_secret(key, value) or place
                saved.append(key)
                break
    names = ", ".join(key.value for key in saved) or "ничего нового"
    where = f"в файле {place}" if place else "в хранилище Windows"
    say(f"Сохранено {where}: {names}")
    return saved


def export_keys(path: Path, *, say: Say = print) -> int:
    """Write the keyring's keys to `path` as a keys file, to carry them to a server.

    Only the keys' names and their number are said, never a value.
    """
    found = stored_in_key_store()
    if not found:
        say(f"{_FAIL}В хранилище нет ни одного ключа: файл не записан")
        return 1
    write_keys_file(path, found)
    names = ", ".join(key.value for key in found)
    say(f"{_OK}Записал {len(found)} ключей в {path}: {names}")
    return 0


async def login(*, force: bool, client: Any | None = None, say: Say = print) -> None:
    """Log the agent's Telegram account in and keep the session string in the keyring.

    The phone, the code and the 2FA password are asked in the terminal by Telethon. Only the
    account's name is shown. A stored session is kept unless `force`.
    """
    if get_secret(Key.TG_SESSION) and not force:
        say(
            "Сессия уже сохранена. Войти заново: uv run python -m tg_agent.setup login --force"
        )
        return
    if client is None:
        client = _telegram("")
    try:
        await client.start()  # asks for the phone, the code and the 2FA password
        set_secret(Key.TG_SESSION, client.session.save())
        say(f"Вошёл как {_name(await client.get_me())}. Сессия сохранена в хранилище")
    finally:
        await client.disconnect()


async def check(
    settings: Settings,
    *,
    anthropic_client: Any | None = None,
    bot: Any | None = None,
    telegram: Any | None = None,
    say: Say = print,
) -> bool:
    """Is everything the agent needs in place, and accepted where it is used?

    Says what is wrong by the error's kind, never with a secret in it.
    """
    missing = [key.value for key in REQUIRED if not get_secret(key)]
    if missing:
        say(
            f"{_FAIL}Нет секретов: {', '.join(missing)}"
            " (uv run python -m tg_agent.setup secrets, затем login)"
        )
        return False
    results = [
        await _check_anthropic(settings, anthropic_client, say),
        await _check_bot(bot, say),
        await _check_userbot(telegram, say),
    ]
    return all(results)


async def _check_anthropic(settings: Settings, client: Any | None, say: Say) -> bool:
    own = client is None
    if own:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=require_secret(Key.ANTHROPIC_API_KEY))
    try:
        model = await client.models.retrieve(settings.models.decide)
    except Exception as error:  # noqa: BLE001 - every failure is reported by its kind
        say(f"{_FAIL}Anthropic: {type(error).__name__}")
        return False
    finally:
        if own:
            await client.close()
    say(f"{_OK}Anthropic: модель {model.id} доступна")
    return True


async def _check_bot(bot: Any | None, say: Say) -> bool:
    own = bot is None
    try:
        if own:
            from aiogram import Bot

            bot = Bot(require_secret(Key.BOT_TOKEN))
        me = await bot.get_me()
    except Exception as error:  # noqa: BLE001 - every failure is reported by its kind
        say(f"{_FAIL}Бот: {type(error).__name__}")
        return False
    finally:
        if own and bot is not None:
            await bot.session.close()
    say(f"{_OK}Бот: @{me.username}")
    return True


async def _check_userbot(client: Any | None, say: Say) -> bool:
    if client is None:
        client = _telegram(require_secret(Key.TG_SESSION))
    try:
        await client.connect()
        if not await client.is_user_authorized():
            say(
                f"{_FAIL}Юзербот: сессия больше не действует"
                " (uv run python -m tg_agent.setup login --force)"
            )
            return False
        me = await client.get_me()
    except Exception as error:  # noqa: BLE001 - every failure is reported by its kind
        say(f"{_FAIL}Юзербот: {type(error).__name__}")
        return False
    finally:
        await client.disconnect()
    say(f"{_OK}Юзербот: {_name(me)}")
    return True


def _telegram(session: str) -> Any:
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    api_id = require_secret(Key.TG_API_ID)
    return TelegramClient(
        StringSession(session), int(api_id), require_secret(Key.TG_API_HASH)
    )


def _name(me: Any) -> str:
    """The account's name and @username: never its phone number."""
    name = " ".join(part for part in (me.first_name, me.last_name) if part)
    username = f" (@{me.username})" if me.username else ""
    return (name or "без имени") + username


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except MissingSecret as error:  # e.g. a keys file others can read: one plain line
        print(f"{_FAIL}{error}")
        return 1


def _main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tg_agent.setup", description="Настройка ИИ-помощника"
    )
    config = argparse.ArgumentParser(add_help=False)
    config.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.toml",
        help="файл настроек (по умолчанию config.toml в корне проекта)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    secrets = commands.add_parser(
        "secrets", parents=[config], help="ключи и токены в хранилище (или keys.toml)"
    )
    secrets.add_argument("--force", action="store_true", help="заменить сохранённые")
    login_ = commands.add_parser(
        "login", parents=[config], help="вход в Telegram-аккаунт агента"
    )
    login_.add_argument("--force", action="store_true", help="войти заново")
    commands.add_parser("check", parents=[config], help="проверить, что всё на месте")
    export = commands.add_parser(
        "export-keys", help="записать ключи из хранилища Windows в файл для сервера"
    )
    export.add_argument("path", type=Path, help="куда записать keys.toml")
    args = parser.parse_args(argv)
    if args.command == "export-keys":
        return export_keys(args.path)
    try:  # where the keys file is: the data folder the settings name
        use_data_dir(load_settings(args.config).data_dir)
    except ConfigError as error:
        print(f"{_FAIL}Настройки: {error}")
        return 1
    if args.command == "secrets":
        ask_secrets(force=args.force)
        return 0
    if args.command == "login":
        asyncio.run(login(force=args.force))
        return 0
    try:
        settings = load_config(args.config)
    except ConfigError as error:
        print(f"{_FAIL}Настройки: {error}")
        return 1
    print(
        f"{_OK}Настройки: {args.config.name}, владельцев: {len(settings.owner.control_user_ids)}"
    )
    return 0 if asyncio.run(check(settings)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
