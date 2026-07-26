from pathlib import Path
import sys

from bs4 import BeautifulSoup


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

html_path = Path(__file__).resolve().parent / "aapbdbdomjkkjkaonfhkkikfgjllcleb.html"

with open(html_path, encoding="utf-8", errors="ignore") as f:
    tmp = f.read()

soup = BeautifulSoup(tmp, "html.parser")
text = soup.get_text()

print(text)
