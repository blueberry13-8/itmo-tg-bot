import re
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pdfplumber
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma.vectorstores import Chroma

from utils import ensure_dirs, url_slug, load_program_urls, abs_url
from config import PLANS, RAW, CHROMA_DIR


def find_plan_pdf_link(html: str) -> str | None:
    """
    Скачиваем страницу и находим учебный план
    """
    soup = BeautifulSoup(html, "html.parser")
    # 1) Пытаемся найти кнопку/ссылку по тексту
    text_rx = re.compile(r"скачать.*учеб.*план|учебный\s*план", re.I)
    for a in soup.find_all("a"):
        txt = (a.get_text(" ") or "").strip()
        href = a.get("href") or ""
        if text_rx.search(txt) and href.lower().endswith(".pdf"):
            return href
    # 2) Фолбэк: первая PDF на странице
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if href.lower().endswith(".pdf"):
            return href
    return None


def scrape():
    urls = load_program_urls()
    for url in urls:
        slug = url_slug(url)
        print(f"[scrape] {slug}: {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        html = r.text
        (RAW / f"{slug}.html").write_text(html, encoding="utf-8")

        pdf_rel = find_plan_pdf_link(html)
        if not pdf_rel:
            print(f"\t!Не найдена ссылка на PDF учебного плана на {url}")
            continue
        pdf_url = abs_url(url, pdf_rel)
        pr = requests.get(pdf_url, timeout=60)
        pr.raise_for_status()
        pdf_path = PLANS / f"{slug}.pdf"
        pdf_path.write_bytes(pr.content)
        print(f"\tСкачан учебный план: {pdf_path}")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Парсим PDF в текст
    """
    full = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            full.append(page.extract_text() or "")
    return "\n".join(full)


def extract_electives_struct(text: str):
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    print("Lines", len(lines))
    p_header_pool = re.compile(r"^Пул выборных дисциплин\.\s*(\d)\s*семестр", re.I)
    p_header_soft = re.compile(r"^Элективные микромодули\s*Soft\s*Skills", re.I)
    p_break = re.compile(r"^(Блок|Обязательные дисциплины|Универсальная|Магистратура/Аспирантура|Мировоззренческий модуль|Аспирантский трек)", re.I)
    # строка предмета: "2 Технологии и практики MLOps 6 216" или "1, 2, 3 Управление мотивацией 1 36"
    p_course = re.compile(r"^(?P<sem>\d+(?:\s*,\s*\d+)*)\s+(?P<title>.+?)\s+(?P<ze>\d+)\s+(?P<hours>\d+)$")

    active, cur_sem = None, None
    out = []
    for ln in lines:
        m_pool = p_header_pool.search(ln)
        if m_pool:
            active = "pool"
            cur_sem = int(m_pool.group(1))
            continue
        if p_header_soft.search(ln):
            active = "soft"
            cur_sem = None
            continue
        if p_break.search(ln):
            active, cur_sem = None, None
            continue

        m_course = p_course.match(ln)
        if m_course and active in {"pool", "soft"}:
            sems_raw = m_course.group("sem")
            title = m_course.group("title").strip(" .")
            ze = int(m_course.group("ze"))
            hours = int(m_course.group("hours"))
            for s in re.split(r"\s*,\s*", sems_raw):
                try:
                    sem_i = int(s)
                except ValueError:
                    sem_i = cur_sem
                out.append({"title": title, "semester": sem_i, "ze": ze, "hours": hours, "category": active})

    seen, uniq = set(), []
    for d in out:
        key = (d["title"].lower(), d["semester"], d["category"])
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def extract_electives(text: str) -> list[str]:
    """
    Сохраняем совместимость со старым форматом: возвращаем только названия
    (с префиксом семестра), но при необходимости доступна extract_electives_struct.
    """
    items = extract_electives_struct(text)
    print(f"Кол-во элективов {len(items)}")
    return [f"Сем {it['semester']}: {it['title']}" for it in items]


def parse() -> None:
    ensure_dirs()
    for pdf in PLANS.glob("*.pdf"):
        slug = pdf.stem
        text = extract_text_from_pdf(pdf)
        data = {
            "program": slug,
            "pdf": str(pdf),
            "text": text,
            "electives": extract_electives(text),
        }
        out = PLANS / f"{slug}.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[parse] {slug}: сохранен {out}")


def load_docs_from_json() -> list[Document]:
    docs = []
    for jf in PLANS.glob("*.json"):
        j = json.loads(jf.read_text(encoding="utf-8"))
        prog = j["program"]
        # Один документ на программу (простота). Разрежем текст ниже.
        docs.append(Document(page_content=j["text"], metadata={"program": prog, "source": prog}))
    return docs


def index() -> None:
    """
    Строим Chroma для RAG
    """
    docs = load_docs_from_json()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = []
    for d in docs:
        for c in splitter.split_text(d.page_content):
            chunks.append(Document(page_content=c, metadata=d.metadata))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=str(CHROMA_DIR))
    print(f"[index] Готово: {CHROMA_DIR}")


if __name__ == "__main__":
    load_dotenv()
    ensure_dirs()

    scrape()
    parse()
    index()
