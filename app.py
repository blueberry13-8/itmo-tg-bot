import os
import json
import argparse
import difflib
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma.vectorstores import Chroma

from config import PLANS, CHROMA_DIR


def recommend_electives(profile: str, program: str, top_k: int = 5) -> list[str]:
    """
    Рекомендации по элективам на базе интересов
    """
    jf = PLANS / f"{program}.json"
    if not jf.exists():
        return []
    j = json.loads(jf.read_text(encoding="utf-8"))
    items = j.get("electives", [])
    scored = []
    for e in items:
        s = difflib.SequenceMatcher(None, profile.lower(), e.lower()).ratio()
        scored.append((s, e))
    scored.sort(reverse=True)
    return [e for _, e in scored[:top_k]]


def cmd_chat(model: str = "gpt-4o-mini"):
    """
    Чат в консоли
    """
    llm = ChatOpenAI(model=model, temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    # Выбор программы
    progs = sorted({m["program"] for m in vs.get(collection_name=None).get("metadatas", [[{"program":"ai"}]])[0]}) if False else [p.stem for p in PLANS.glob("*.json")]
    if not progs:
        print("Нет распарсенных программ. Выполните scrape → parse → index.")
        return
    print("Доступные программы:", ", ".join(progs))
    selected = input("Выберите программу (введите слаг, напр. ai): ").strip() or progs[0]

    # Ретривер с фильтром по программе и порогом релевантности
    retriever = vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.2, "filter": {"program": selected}},
    )

    system_msg = (
        "Ты помощник абитуриента. Отвечай только на вопросы по обучению и содержанию выбранной магистратуры. "
        "Если вопрос вне темы — вежливо откажись и предложи спросить про программу, учебный план, предметы, сроки, траектории, дисциплины."
    )

    print("\nНачинаем диалог. Подсказка: напишите 'рекомендации: ваш бэкграунд/интересы' для подбора элективов. 'exit' — выход.\n")
    while True:
        q = input("Вы: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "выход"}:
            break
        # Роутинг: если пользователь просит рекомендации
        if q.lower().startswith("рекомендации:") or ("электив" in q.lower() and any(w in q.lower() for w in ["совет", "подбери", "рекоменд"])):
            profile = q.split(":", 1)[1] if ":" in q else q
            recs = recommend_electives(profile, selected)
            if recs:
                print("Бот: Возможные элективы (на основе похожести к вашим интересам):\n- " + "\n- ".join(recs))
            else:
                print("Бот: Не нашел элективов в плане. Попробуйте уточнить интересы или выбрать другую программу.")
            continue

        # Иначе — обычный RAG QA
        docs = retriever.invoke(q)
        if not docs:
            print("Бот: Это вне моей компетенции. Задавайте вопросы про обучение в выбранной магистратуре (учебный план, дисциплины, сроки и т.п.).")
            continue

        # Простой prompt + контекст
        context = "\n\n".join(d.page_content[:1200] for d in docs)
        messages = [
            ("system", system_msg),
            ("user", f"Вопрос: {q}\n\nКонтекст (фрагменты из документов):\n{context}\n\nКоротко и по делу.")
        ]
        # Мини-обертка вокруг ChatOpenAI без LangChain Chain, чтобы код был короче
        resp = llm.invoke(messages)
        print("Бот:", resp.content.strip())

# --- Шаг 6. Telegram-бот (опционально, максимально простой) ---

def cmd_telegram(model: str = "gpt-4o-mini"):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Нет TELEGRAM_BOT_TOKEN в .env")
        return
    from aiogram import Bot, Dispatcher
    from aiogram.types import Message
    from aiogram.filters import CommandStart
    import asyncio

    llm = ChatOpenAI(model=model, temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    # Возьмем первую доступную программу как дефолт
    programs = [p.stem for p in PLANS.glob("*.json")] or ["ai"]
    DEFAULT_PROG = programs[0]

    async def build_retriever(prog):
        return vs.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.2, "filter": {"program": prog}},
        )

    system_msg = (
        "Ты помощник абитуриента. Отвечай только по программам ИТМО. Вне темы — вежливо откажись."
    )

    bot = Bot(token)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start(m: Message):
        await m.answer("Привет! Я помогу с вопросами по магистратурам ИТМО. Напишите: /prog ai или /prog ai_product, затем задавайте вопросы. Для рекомендаций по элективам напишите: рекомендации: ваш бэкграунд")

    @dp.message()
    async def all_messages(m: Message):
        text = m.text or ""
        # Выбор программы простой командой
        if text.startswith("/prog "):
            prog = text.split(" ", 1)[1].strip()
            dp.workflow_data["prog"] = prog
            await m.answer(f"Ок, работаем с программой: {prog}")
            return

        prog = dp.workflow_data.get("prog", DEFAULT_PROG)
        if text.lower().startswith("рекомендации:"):
            profile = text.split(":", 1)[1] if ":" in text else text
            recs = recommend_electives(profile, prog)
            if recs:
                await m.answer("Возможные элективы:\n- " + "\n- ".join(recs))
            else:
                await m.answer("Не нашел элективов в плане. Уточните интересы или смените программу: /prog ...")
            return

        retriever = await build_retriever(prog)
        docs = retriever.get_relevant_documents(text)
        if not docs:
            await m.answer("Это вне моей компетенции. Спросите про программу, учебный план, дисциплины, сроки и т.п.")
            return
        context = "\n\n".join(d.page_content[:1000] for d in docs)
        messages = [
            ("system", system_msg),
            ("user", f"Вопрос: {text}\n\nКонтекст:\n{context}\n\nКоротко ответь.")
        ]
        resp = llm.invoke(messages)
        await m.answer(resp.content.strip())

    asyncio.run(dp.start_polling(bot))


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="ITMO chat-bot minimal")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="Локальный чат в терминале")
    p_chat.add_argument("--model", default="gpt-4o-mini")

    p_tg = sub.add_parser("telegram", help="Запустить Telegram-бота")
    p_tg.add_argument("--model", default="gpt-4o-mini")

    args = parser.parse_args()

    if args.cmd == "chat":
        cmd_chat(model=args.model)
    elif args.cmd == "telegram":
        cmd_telegram(model=args.model)
