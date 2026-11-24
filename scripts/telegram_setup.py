"""Liga o bot do Telegram ao projeto: guarda o token, descobre o chat e manda um teste.

Antes de rodar:
  1. Crie o bot no @BotFather (/newbot) e copie o token.
  2. Abra a conversa com o seu bot e aperte "Iniciar" (ou adicione o bot a um
     grupo e mande /start@nome_do_bot lá, para os alertas irem para o grupo).

Uso:
  python scripts/telegram_setup.py              # pede o token na primeira vez
  python scripts/telegram_setup.py --reset      # troca o token salvo
  python scripts/telegram_setup.py --chat-id N  # usa um chat específico
"""
import argparse
import getpass
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.interface.telegram import telegram_api  # noqa: E402
from src.utils.config import SECRETS_FILE, load_secrets  # noqa: E402


def save_secrets(token: str, chat_id) -> None:
    SECRETS_FILE.write_text(
        "# Credenciais do bot do Telegram. Este arquivo NÃO vai para o git (.gitignore).\n"
        + yaml.safe_dump({"telegram": {"token": token, "chat_id": chat_id}}, allow_unicode=True),
        encoding="utf-8")


def find_chats(token: str) -> list[dict]:
    """Conversas e grupos que falaram com o bot, do mais recente ao mais antigo."""
    chats: dict[int, dict] = {}
    for update in telegram_api(token, "getUpdates"):
        source = update.get("message") or update.get("my_chat_member") or {}
        chat = source.get("chat")
        if chat:
            chats.pop(chat["id"], None)  # reinsere para ficar na ordem do mais recente
            chats[chat["id"]] = chat
    return list(reversed(chats.values()))


def describe(chat: dict) -> str:
    if chat["type"] == "private":
        return f"conversa com {chat.get('first_name', '')} {chat.get('last_name', '')}".strip()
    return f"grupo \"{chat.get('title', '')}\""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="pede um token novo")
    parser.add_argument("--chat-id", help="usa este chat em vez de descobrir sozinho")
    args = parser.parse_args()

    token = "" if args.reset else load_secrets().get("telegram", {}).get("token", "")
    if not token:
        print("Cole o token do @BotFather e aperte Enter (ele não aparece na tela, é normal).")
        token = getpass.getpass("Token: ").strip()
    if not re.fullmatch(r"\d+:[\w-]+", token):
        sys.exit("Isso não parece um token. Ele tem o formato 123456789:AAH... (rode de novo com --reset)")

    try:
        bot = telegram_api(token, "getMe")
    except RuntimeError as error:
        sys.exit(f"Não consegui usar esse token: {error}. Rode de novo com --reset.")
    print(f"Bot encontrado: @{bot['username']}")

    if args.chat_id:
        chat_id = int(args.chat_id) if args.chat_id.lstrip("-").isdigit() else args.chat_id
    else:
        chats = find_chats(token)
        if not chats:
            save_secrets(token, "")
            sys.exit(f"Token salvo, mas o bot ainda não recebeu nenhuma mensagem.\n"
                     f"Abra https://t.me/{bot['username']}, aperte Iniciar e rode este script de novo.")
        chat_id = chats[0]["id"]
        print(f"Os alertas vão para: {describe(chats[0])}")
        for other in chats[1:]:
            print(f"  (também encontrei {describe(other)}; para usá-lo: --chat-id {other['id']})")

    save_secrets(token, chat_id)
    telegram_api(token, "sendMessage", json={
        "chat_id": chat_id,
        "text": "✅ Monitor de Motorista conectado. Os alertas vão chegar aqui."})
    print(f"Configuração salva em {SECRETS_FILE.name}. Mensagem de teste enviada: confira o Telegram!")


if __name__ == "__main__":
    main()
