"""Export poster.html -> poster.pdf (A0, one page) + poster.png thumbnail + full-res print PNG.
Replaces the skill's render_poster.py, which fails on Windows (os.fsync on the font sync)."""
import pathlib, sys
from playwright.sync_api import sync_playwright
from PIL import Image
here = pathlib.Path(__file__).resolve().parent.parent.parent
html = here / "poster.html"
url = "file:///" + str(html).replace("\\", "/")
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 3178, "height": 4493})
    pg.emulate_media(media="print")
    pg.goto(url); pg.wait_for_timeout(4000)
    pg.pdf(path=str(here / "poster.pdf"), width="33.1in", height="46.8in", print_background=True,
           prefer_css_page_size=True, page_ranges="1",
           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    pg.screenshot(path=str(here / "assets/meta/poster_print_full.png"), full_page=False)
    b.close()
im = Image.open(here / "assets/meta/poster_print_full.png"); im.thumbnail((1589, 2247)); im.save(here / "poster.png")
import pymupdf
d = pymupdf.open(str(here / "poster.pdf"))
print("pdf pages", len(d), "size_in", round(d[0].rect.width / 72, 2), round(d[0].rect.height / 72, 2), "| png", im.size)
