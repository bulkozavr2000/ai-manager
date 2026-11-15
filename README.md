# tg-agent — ИИ-помощник в Telegram

Личный агент, который ведёт переписку в Telegram от имени владельца и честно представляется
ИИ-помощником. Владелец пишет задачу обычными словами («узнай, сколько стоит печать 5 корпусов
из PETG, бюджет до 300 000») в свой управляющий бот; агент сам находит подходящих людей, пишет
им со своего рабочего аккаунта, уточняет цену, сроки и условия, а владельцу присылает короткие
карточки с итогом и кнопками. Решения, которые стоят денег (заказ, выбор исполнителя), всегда
остаются за владельцем.

Модели — Claude (Anthropic API). Язык переписки — русский и узбекский. Python 3.12, Docker.

> Значения по умолчанию описывают вымышленного владельца «Саид Ахмедов» из Ташкента: так
> устроены промпты, тесты и eval-сценарии. Свои имя и город задайте в `config.toml`.

## Как это устроено

Программа — один процесс на asyncio с базой SQLite (`data/agent.db`).

- **Управляющий бот** (aiogram, `tg_agent.control`). Через него владелец ставит задачи,
  отвечает на вопросы агента и жмёт кнопки на карточках: «Запустить», «Продолжить»,
  «Подтвердить заказ?», «Дать твой телефон?». Слушается только Telegram-id из `owner.control_user_ids`.
- **Секретарь** (`tg_agent.agent.secretary`). Разбирает сообщения владельца: новая задача,
  уточнение к старой, «запомни», «не работай больше с …», вопрос о статусе.
- **Агент задачи** (`tg_agent.agent`). На каждое событие (ответ собеседника, таймер, кнопка)
  «просыпается», видит задачу, переписку и память и вызывает инструменты: написать человеку,
  спросить владельца, найти людей, завершить задачу. Число шагов и деньги на одно пробуждение
  ограничены.
- **Поиск людей** (`tg_agent.search`). Веб-поиск, поиск в Telegram, OLX и любые сайты через
  собственный браузер агента (Playwright). Если сайт просит войти, бот присылает ссылку, по
  которой владелец входит с телефона в удалённую вкладку.
- **Рабочий аккаунт** (Telethon, `tg_agent.gateway`). Обычный пользовательский аккаунт
  Telegram, от имени которого идёт переписка. Сообщения уходят с паузами, только в рабочие
  часы и с ограничением новых контактов в день.
- **Проверки** (`tg_agent.policy`). До модели — фильтры входящих; после модели — DLP и
  отдельная модель-«судья», которая смотрит каждый черновик перед отправкой.

## Запуск

Нужны: ключ Anthropic API, токен бота от @BotFather, `api_id` и `api_hash` с my.telegram.org
для рабочего аккаунта и сам рабочий аккаунт Telegram (лучше отдельный, не личный).

### Без Docker

```bash
uv sync
uv run playwright install chromium          # или поставьте search.browser = "msedge"/"chrome"
cp config.example.toml config.toml          # и впишите свои owner.full_name, control_user_ids
uv run python -m tg_agent.setup secrets     # ключи — в хранилище ОС (keyring)
uv run python -m tg_agent.setup login       # вход в рабочий аккаунт
uv run python -m tg_agent.setup check       # всё ли на месте
uv run python -m tg_agent
```

### В Docker

В контейнере нет хранилища ключей ОС, поэтому ключи лежат в `data/keys.toml` (только для
владельца файла: `chmod 600`).

```bash
cp config.example.toml config.toml          # search.browser = "chromium": на сервере нет Edge
mkdir -p data && sudo chown -R 1000:1000 data
docker compose run --rm agent python -m tg_agent.setup secrets
docker compose run --rm agent python -m tg_agent.setup login
docker compose up -d --build
docker compose logs -f
```

Готовый образ публикуется в GitHub Container Registry при теге `v*`
(`ghcr.io/<владелец>/<репозиторий>`).

## Тесты

```bash
uv sync
uv run pytest -q
```

Тесты не ходят в сеть и не тратят деньги: модели и Telegram подменены фейками, хранилище
ключей — в памяти. Тесты браузера пропускаются, если браузера нет (`TG_AGENT_TEST_BROWSER=chromium`
выбирает Chromium от Playwright). Живые тесты с настоящим API запускаются только с
`TG_AGENT_LIVE=1`. Платные прогоны сценариев: `uv run python -m tg_agent.evals` (спросит
подтверждение стоимости).

## Модель безопасности

- Агент всегда говорит, что он ИИ-помощник, и никогда не выдаёт себя за человека.
- Он не подтверждает заказ, оплату или предоплату сам: только после кнопки владельца.
- Телефон и адрес владельца хранятся отдельно от настроек и уходят собеседнику только
  после кнопки владельца; DLP не пропускает чужие ссылки, номера карт и
  прочие личные данные, модель-судья проверяет каждый черновик.
- Текст собеседников и сайтов считается недоверенным: он экранируется и не может отдавать
  агенту команды. Браузер агента не открывает адреса этого компьютера и локальной сети.
- Лимиты расходов (день, месяц, задача, одно пробуждение), лимит новых контактов в день,
  рабочие и тихие часы.
- Секреты — только в хранилище ОС или в `data/keys.toml`; `config.toml`, `data/` и сессии
  не попадают в git и в образ.

---

## English summary

**tg-agent** is a personal Telegram assistant powered by Claude. The owner describes a task in
plain words to a control bot; the agent finds suitable people (web, Telegram, OLX, any site via
its own Playwright browser), writes to them from a separate Telegram user account (Telethon),
always introducing itself as an AI assistant, negotiates price, timing and terms, and reports
back with short cards and buttons. Anything that costs money stays the owner's decision.

- **Run without Docker:** `uv sync`, `uv run playwright install chromium`, copy
  `config.example.toml` to `config.toml`, then `uv run python -m tg_agent.setup secrets`,
  `... setup login`, `... setup check`, and `uv run python -m tg_agent`.
- **Run with Docker:** keys live in `data/keys.toml`; `docker compose run --rm agent python -m
  tg_agent.setup secrets`, `... setup login`, then `docker compose up -d --build`. Tagging
  `v*` publishes an image to GitHub Container Registry.
- **Tests:** `uv run pytest -q` — fully offline (fake models, fake Telegram, in-memory key
  store); live tests need `TG_AGENT_LIVE=1`.
- **Safety:** honest AI disclosure, no orders or payments without the owner's button, DLP plus
  an LLM judge on every outgoing draft, untrusted text is escaped and never obeyed, spend caps,
  daily new-contact limits, working and quiet hours, secrets only in the OS keyring or
  `data/keys.toml`.

Defaults describe a fictional owner; set your own name and city in `config.toml`. The prompts
and conversations are in Russian and Uzbek.

## Лицензия / License

MIT — см. файл [LICENSE](LICENSE). MIT, see [LICENSE](LICENSE).
