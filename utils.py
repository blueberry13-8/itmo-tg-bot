import os
from urllib.parse import urlparse
import requests

from config import CHROMA_DIR, DATA, DEFAULT_URLS, PLANS, RAW


def ensure_dirs() -> None:
    for p in [DATA, RAW, PLANS, CHROMA_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def url_slug(url: str) -> str:
    # Берем последний сегмент URL как слаг
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] or "program"


def load_program_urls() -> list[str]:
    env_urls = os.getenv("PROGRAM_URLS")
    if env_urls:
        return [u.strip() for u in env_urls.split(",") if u.strip()]
    return DEFAULT_URLS


def abs_url(base: str, href: str) -> str:
    # Делает ссылку абсолютной (на случай относительных путей)
    return requests.compat.urljoin(base, href)
