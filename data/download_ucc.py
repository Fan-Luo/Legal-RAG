import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://www.law.cornell.edu"
ARTICLES = ["1", "2", "2A", "3", "4", "4A", "5", "6", "7", "8", "9"]
OUT_DIR = "raw/ucc"
SLEEP_SEC = 0.5

session = requests.Session()
session.headers.update({"User-Agent": "ucc-downloader/1.0"})

def fetch(url: str) -> str:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def _clean_text(text: str) -> str:
  lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
  return "\n".join([ln for ln in lines if ln])

def _clean_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])

def _should_break(prev: str, nxt: str) -> bool:
    if not prev or not nxt:
        return True
    if re.search(r"[。！？；：:]$", prev):
        return True
    if re.match(r"^§\s*\d", nxt):
        return True
    if re.match(r"^[（(]?[0-9一二三四五六七八九十]+[)）\.、]", nxt):
        return True
    if re.match(r"^第[一二三四五六七八九十百千万〇零0-9]+条", nxt):
        return True
    return False

def _merge_lines(text: str) -> str:
    out = []
    cur = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if cur:
                out.append(cur)
                cur = ""
            continue
        if not cur:
            cur = line
            continue
        if _should_break(cur, line):
            out.append(cur)
            cur = line
        else:
            if cur.endswith("-") and len(cur) > 1 and cur[-2].isalpha():
                cur = cur[:-1] + line
            elif re.match(r"^[,.;:)\]]", line):
                cur = cur + line
            elif cur.endswith(("(", "[")):
                cur = cur + line
            else:
                cur = cur + " " + line
    if cur:
        out.append(cur)
    return "\n".join(out).strip()

def extract_content_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", {"id": "content"})
    if content is None:
        return ""
    for nav in content.find_all("nav"):
        nav.decompose()
    for tag in content(["script", "style", "noscript"]):
        tag.decompose()
    text = content.get_text("\n").replace("\xa0", " ")
    text = _clean_text(text)
    return _merge_lines(text)

def section_prefix(article: str) -> str:
    return f"/ucc/{article}/{article}-"

def collect_toc_groups(article_url: str, article: str):
    html = fetch(article_url)
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", {"id": "content"}) or soup
    prefix = section_prefix(article)

    groups = []
    for h2 in content.find_all("h2"):
        title = _clean_text(h2.get_text("\n"))
        if not title:
            continue
        links = []
        for sib in h2.find_next_siblings():
            if sib.name == "h2":
                break
            for a in sib.select("a[href]"):
                href = urljoin(BASE, a["href"])
                if urlparse(href).path.startswith(prefix):
                    links.append(href)
        if links:
            # de-dup while preserving order
            seen = set()
            ordered = []
            for link in links:
                if link not in seen:
                    ordered.append(link)
                    seen.add(link)
            groups.append((title, ordered))
    return groups

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for art in ARTICLES:
        article_url = f"{BASE}/ucc/{art}"
        print(f"[article] {article_url}")
        out_path = os.path.join(OUT_DIR, f"ucc_{art}.txt")

        blocks = []
        for h2_title, links in collect_toc_groups(article_url, art):
            blocks.append(h2_title)  # keep directory page <h2>
            for link in links:
                print(f"  [section] {link}")
                html = fetch(link)
                text = extract_content_text(html)
                if text:
                    blocks.append(text)  # keep child page title (e.g., § 1-101...)
                time.sleep(SLEEP_SEC)
            blocks.append("")  # blank line between h2 groups

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join([b for b in blocks if b is not None]))

if __name__ == "__main__":
    main()
