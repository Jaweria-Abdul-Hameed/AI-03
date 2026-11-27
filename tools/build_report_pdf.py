"""Build the formatted A4 report PDF from the Markdown source.

Usage (from the repository root):
    uv run --no-project --python 3.12 --with markdown --with beautifulsoup4 --with pypdf python tools/build_report_pdf.py

Pipeline: Markdown -> styled HTML (cover, contents, lists of figures/tables, figure and
table captions, Q&A boxes) -> PDF via headless Microsoft Edge (or Chrome). The page numbers
in the contents are read back from the first render and written into a second render.
"""
from __future__ import annotations

import copy
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "my_project" / "report"
MD_FILE = REPORT_DIR / "Superhero_Intelligence_System_Report.md"
PDF_FILE = REPORT_DIR / "Superhero_Intelligence_System_Report.pdf"
BUILD_HTML = REPORT_DIR / "_build_report.html"

TITLE = "Superhero Intelligence System"
SUBTITLE = "Can six ability scores tell a hero from a villain?"
MEMBERS = [("Mohsin Khan", "24i-3135"), ("Jaweria Abdul Hameed", "24i-3025")]
DEADLINE = "06 December 2026"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium",
]

CSS = r"""
@page {
  size: A4;
  margin: 21mm 17mm 19mm 17mm;
  @top-left   { content: "Superhero Intelligence System — Project Report"; font: 7.6pt "Segoe UI", Arial, sans-serif; color: #6B7686; vertical-align: bottom; padding-bottom: 3mm; border-bottom: 0.5pt solid #D5DBE3; }
  @top-right  { content: "Machine Learning · Assignment 03"; font: 7.6pt "Segoe UI", Arial, sans-serif; color: #6B7686; vertical-align: bottom; padding-bottom: 3mm; border-bottom: 0.5pt solid #D5DBE3; }
  @bottom-left  { content: "Mohsin Khan (24i-3135) · Jaweria Abdul Hameed (24i-3025)"; font: 7.6pt "Segoe UI", Arial, sans-serif; color: #6B7686; vertical-align: top; padding-top: 3mm; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 7.6pt "Segoe UI", Arial, sans-serif; color: #6B7686; vertical-align: top; padding-top: 3mm; }
}
@page cover {
  margin: 0;
  @top-left { content: none; border: none; } @top-right { content: none; border: none; }
  @bottom-left { content: none; } @bottom-right { content: none; }
}
:root { --ink: #1B2230; --muted: #5B6575; --navy: #16365C; --accent: #1F5FBF; --rule: #D5DBE3; --zebra: #F5F7FA; --soft: #EEF3FA; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.6pt; line-height: 1.5; color: var(--ink); margin: 0; }
p { margin: 0 0 6pt; text-align: left; }
code { font-family: Consolas, "Courier New", monospace; font-size: 0.86em; background: #F1F3F6; padding: 0 2px; border-radius: 2px; color: #24324A; }
strong { font-weight: 600; color: #121826; }

/* ---------- cover ---------- */
.cover { page: cover; height: 297mm; position: relative; overflow: hidden; break-after: page; }
.cover .band { position: absolute; left: 0; top: 0; bottom: 0; width: 13mm; background: var(--navy); }
.cover .band::after { content: ""; position: absolute; left: 13mm; top: 0; bottom: 0; width: 2.2mm; background: var(--accent); }
.cover .inner { position: absolute; left: 36mm; right: 24mm; top: 46mm; }
.cover .kicker { font-size: 9pt; letter-spacing: 0.22em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
.cover h1 { font-size: 33pt; line-height: 1.08; margin: 7mm 0 4mm; color: var(--navy); font-weight: 700; letter-spacing: -0.01em; }
.cover .sub { font-size: 14.5pt; color: var(--muted); font-style: italic; margin-bottom: 11mm; }
.cover .doc { font-size: 12pt; font-weight: 600; color: var(--ink); padding-top: 5mm; border-top: 1.2pt solid var(--navy); width: 78mm; }
.cover .meta { position: absolute; left: 36mm; right: 24mm; top: 150mm; }
.cover .meta h3 { font-size: 8.4pt; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); margin: 0 0 3mm; font-weight: 600; }
.cover table.members { width: 112mm; border-collapse: collapse; margin-bottom: 12mm; font-size: 11pt; }
.cover table.members th { text-align: left; font-size: 8pt; color: var(--muted); font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; border-bottom: 1pt solid var(--navy); padding: 0 0 2mm; background: none; }
.cover table.members td { padding: 2.6mm 0; border-bottom: 0.5pt solid var(--rule); }
.cover table.members td:last-child { font-family: Consolas, monospace; color: var(--navy); font-weight: 600; }
.cover dl { display: grid; grid-template-columns: 38mm 1fr; row-gap: 2.4mm; margin: 0; font-size: 9.8pt; }
.cover dt { color: var(--muted); } .cover dd { margin: 0; color: var(--ink); }
.cover .foot { position: absolute; left: 36mm; right: 24mm; bottom: 22mm; font-size: 8.2pt; color: var(--muted); border-top: 0.5pt solid var(--rule); padding-top: 3mm; }

/* ---------- contents ---------- */
.front { break-after: page; }
.front h2 { break-before: auto; }
.toc { list-style: none; padding: 0; margin: 0 0 9mm; }
.toc li { display: flex; align-items: baseline; margin: 0; }
.toc li a { color: inherit; text-decoration: none; }
.toc li .dots { flex: 1; border-bottom: 0.9pt dotted #9AA4B2; margin: 0 2mm; transform: translateY(-2.5pt); }
.toc li .pg { font-variant-numeric: tabular-nums; min-width: 7mm; text-align: right; color: var(--navy); font-weight: 600; }
.toc .l2 { font-weight: 600; font-size: 10.2pt; margin-top: 3.4mm; color: var(--navy); }
.toc .l3 { padding-left: 6mm; font-size: 9.3pt; margin-top: 1.1mm; }
.lists h3 { margin-top: 2mm; }
.lists .toc { margin-bottom: 6mm; } .lists .toc li { font-size: 9pt; margin-top: 1.1mm; }
.lists .toc li b { color: var(--navy); font-weight: 600; width: 12mm; display: inline-block; }

/* ---------- headings ---------- */
h2 { font-size: 16pt; color: var(--navy); margin: 0 0 5mm; padding-bottom: 2.4mm; border-bottom: 1.6pt solid var(--navy); break-before: page; break-after: avoid; font-weight: 700; letter-spacing: -0.005em; }
h2 .marks { float: right; font-size: 9pt; font-weight: 600; color: #fff; background: var(--accent); border-radius: 9pt; padding: 1.2pt 8pt; margin-top: 4pt; letter-spacing: 0; }
h3 { font-size: 11.6pt; color: var(--navy); margin: 7mm 0 3mm; break-after: avoid; font-weight: 600; }
h3::before { content: ""; display: inline-block; width: 2.4mm; height: 2.4mm; background: var(--accent); margin-right: 2.4mm; transform: translateY(-1pt); }
ul, ol { margin: 0 0 6pt; padding-left: 5.5mm; } li { margin: 0 0 3pt; }

/* ---------- tables ---------- */
.tbl { margin: 4mm 0 5mm; break-inside: avoid; }
.tbl .title { font-size: 8.8pt; color: var(--ink); margin-bottom: 1.6mm; break-after: avoid; }
.tbl .title b { color: var(--navy); }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; line-height: 1.35; }
thead th { background: var(--navy); color: #fff; font-weight: 600; padding: 2.1mm 2mm; text-align: left; border: none; vertical-align: bottom; }
tbody td { padding: 1.5mm 2mm; border-bottom: 0.5pt solid var(--rule); vertical-align: top; }
tbody tr:nth-child(even) td { background: var(--zebra); }
tbody tr:last-child td { border-bottom: 1pt solid var(--navy); }
td code, th code { background: none; padding: 0; font-size: 0.95em; }
th code { color: #fff; }
h4 { font-size: 10pt; color: var(--navy); margin: 5mm 0 2mm; }
th[style*="right"], td[style*="right"] { font-variant-numeric: tabular-nums; }
.tbl.wide table { font-size: 7.1pt; } .tbl.wide thead th { padding: 1.6mm 1.3mm; } .tbl.wide tbody td { padding: 1.2mm 1.3mm; }
.split { display: grid; grid-template-columns: 1fr 1fr; column-gap: 5mm; }
.split table { font-size: 7.1pt; } .split thead th { padding: 1.3mm 1.4mm; } .split tbody td { padding: 0.62mm 1.4mm; }
.note { font-size: 8.5pt; color: #3F4857; background: var(--soft); border-left: 2.2pt solid var(--accent); padding: 2.4mm 3.2mm; margin: 0 0 6mm; text-align: left; break-inside: avoid; }
.note b { color: var(--navy); }

/* ---------- figures ---------- */
figure { margin: 4mm auto 6mm; break-inside: avoid; text-align: center; }
figure img { display: block; margin: 0 auto 2.4mm; border: 0.5pt solid var(--rule); }
figcaption { font-size: 8.5pt; color: #3F4857; text-align: left; line-height: 1.45; max-width: 172mm; margin: 0 auto; }
figcaption b { color: var(--navy); }

/* ---------- Q&A ---------- */
.qa { border: 0.6pt solid var(--rule); border-left: 2.6pt solid var(--navy); border-radius: 1.5mm; padding: 2.8mm 3.6mm 2.2mm; margin: 0 0 3.2mm; break-inside: avoid; background: #FBFCFE; }
.qa .q { font-weight: 600; color: var(--navy); margin-bottom: 1.4mm; font-size: 9.6pt; }
.qa .q .tag { display: inline-block; background: var(--navy); color: #fff; border-radius: 1mm; padding: 0 5pt; margin-right: 6pt; font-size: 8.6pt; }
.qa .a { font-size: 9.1pt; } .qa .a p { margin-bottom: 3pt; } .qa ol { margin: 2pt 0 2pt; }
.summary ul { list-style: none; padding: 0; margin-bottom: 4mm; } .summary li { padding: 2.6mm 3.4mm; border: 0.6pt solid var(--rule); border-radius: 1.5mm; margin-bottom: 2.4mm; background: #FBFCFE; }
"""


def png_aspect(path: Path) -> float:
    w, h = struct.unpack(">II", path.read_bytes()[16:24])
    return w / h


def clean_text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text()).strip()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def build_body(md_text: str, page_map: dict[str, int] | None) -> str:
    soup = BeautifulSoup(markdown.markdown(md_text, extensions=["tables", "sane_lists"]), "html.parser")

    # Drop the Markdown title block (the cover page replaces it) and horizontal rules.
    soup.find("h1").decompose()
    first_p = soup.find("p")
    if first_p and "Group Members" in first_p.get_text():
        first_p.decompose()
    for hr in soup.find_all("hr"):
        hr.decompose()

    toc, figs, tabs = [], [], []

    # Headings: ids, marks badges, contents entries.
    for h in soup.find_all(["h2", "h3"]):
        text = clean_text(h)
        h["id"] = "s-" + slug(text)
        if h.name == "h2":
            m = re.search(r"\((\d+) Marks\)", text)
            if m:
                for node in list(h.contents):
                    if isinstance(node, NavigableString) and m.group(0) in node:
                        node.replace_with(node.replace(" " + m.group(0), "").replace(m.group(0), ""))
                badge = soup.new_tag("span", attrs={"class": "marks"}); badge.string = f"{m.group(1)} marks"
                h.insert(0, badge)
            toc.append(("l2", h["id"], re.sub(r"\s*\(\d+ Marks\)", "", text)))
        else:
            toc.append(("l3", h["id"], text))
    summary_h2 = soup.find("h2")
    if summary_h2 and "Executive Summary" in summary_h2.get_text():
        node = summary_h2.find_next_sibling()
        while node is not None and node.name != "h2":
            nxt = node.find_next_sibling()
            if node.name == "ul":
                node.wrap(soup.new_tag("div", attrs={"class": "summary"}))
            node = nxt

    # Tables: h4 "Table Tn: title" -> caption block above the table; caption paragraph -> note below.
    for h4 in soup.find_all("h4"):
        m = re.match(r"Table (T\d+[a-z]?):\s*(.*)", clean_text(h4))
        table = h4.find_next_sibling()
        if not m or table is None or table.name != "table":
            continue
        tid, ttitle = m.groups()
        box = soup.new_tag("div", attrs={"class": "tbl", "id": "t-" + tid})
        title = soup.new_tag("div", attrs={"class": "title"})
        b = soup.new_tag("b"); b.string = f"Table {tid}."
        title.append(b)
        rest = copy.copy(h4); rest.name = "span"
        # strip the "Table Tn:" prefix while keeping inline code
        first = rest.contents[0]
        rest.contents[0].replace_with(NavigableString(" " + re.sub(r"^Table T\d+[a-z]?:\s*", "", str(first))))
        title.append(rest)
        h4.replace_with(box)
        box.append(title)
        n_cols = len(table.find("tr").find_all(["th", "td"]))
        if n_cols >= 9:
            box["class"] = box["class"] + ["wide"]
        if tid == "T1c":
            box.append(split_column_table(soup, table))
            table.decompose()
        else:
            box.append(table.extract())
        tabs.append((tid, box["id"], ttitle))

    for p in soup.find_all("p"):
        em = p.find("em")
        if not em or not em.get_text().startswith("Caption ("):
            continue
        m = re.match(r"Caption \((Graph|Table) (\w+)\):", em.get_text())
        kind, ref = m.groups()
        em.decompose()
        if kind == "Table":
            p.name = "div"; p["class"] = "note"
            lead = soup.new_tag("b"); lead.string = f"Table {ref}. "
            p.insert(0, lead)
            prev = p.find_previous_sibling()
            if prev is not None and "tbl" in (prev.get("class") or []):
                prev.append(p.extract())  # keep the note on the same page as its table

    # Figures: <p><img></p> + caption paragraph -> <figure>.
    for img in soup.find_all("img"):
        p = img.parent
        alt = img.get("alt", "")
        m = re.match(r"(G\d+):\s*(.*)", alt)
        gid, gtitle = m.groups()
        cap = p.find_next_sibling("p")
        fig = soup.new_tag("figure", attrs={"id": "f-" + gid})
        aspect = png_aspect((REPORT_DIR / img["src"]).resolve())
        width_mm = min(172.0, 96.0 * aspect)
        new_img = soup.new_tag("img", attrs={"src": img["src"], "alt": alt, "style": f"width:{width_mm:.1f}mm"})
        fig.append(new_img)
        fc = soup.new_tag("figcaption")
        b = soup.new_tag("b"); b.string = f"Figure {gid} — {gtitle}. "
        fc.append(b)
        if cap is not None:
            for node in list(cap.contents):
                fc.append(node.extract())
            cap.decompose()
        fig.append(fc)
        p.replace_with(fig)
        figs.append((gid, fig["id"], gtitle))

    # Q&A lists -> answer boxes.
    for h3 in soup.find_all("h3"):
        if "Report Answers" not in h3.get_text():
            continue
        ul = h3.find_next_sibling("ul")
        for li in ul.find_all("li", recursive=False):
            strong = li.find("strong")
            m = re.match(r"(Q\d+)\s*\((.*)\):?\s*$", clean_text(strong))
            qid, qtext = m.groups()
            strong.decompose()
            box = soup.new_tag("div", attrs={"class": "qa"})
            q = soup.new_tag("div", attrs={"class": "q"})
            tag = soup.new_tag("span", attrs={"class": "tag"}); tag.string = qid
            q.append(tag); q.append(qtext.rstrip(":"))
            a = soup.new_tag("div", attrs={"class": "a"})
            for node in list(li.contents):
                a.append(node.extract())
            if a.contents and isinstance(a.contents[0], NavigableString):
                a.contents[0].replace_with(NavigableString(a.contents[0].lstrip()))
            box.append(q); box.append(a)
            ul.insert_before(box)
        ul.decompose()

    front = cover_html() + contents_html(toc, figs, tabs, page_map)
    return front + str(soup)


def split_column_table(soup: BeautifulSoup, table) -> object:
    """Render the 83-row column listing as two side-by-side tables with one shared dtype column."""
    rows = [[c.get_text(strip=True) for c in tr.find_all("td")] for tr in table.find("tbody").find_all("tr")]
    assert all(r[2] == r[4] for r in rows), "train/test dtypes differ — cannot merge dtype columns"
    half = (len(rows) + 1) // 2
    grid = soup.new_tag("div", attrs={"class": "split"})
    for chunk in (rows[:half], rows[half:]):
        t = BeautifulSoup(
            "<table><thead><tr><th style='text-align:right'>#</th><th>Column</th><th>Dtype</th>"
            "<th style='text-align:right'>Train missing</th><th style='text-align:right'>Test missing</th></tr></thead><tbody>"
            + "".join(
                f"<tr><td style='text-align:right'>{r[0]}</td><td><code>{r[1]}</code></td><td><code>{r[2]}</code></td>"
                f"<td style='text-align:right'>{r[3]}</td><td style='text-align:right'>{r[5]}</td></tr>"
                for r in chunk)
            + "</tbody></table>", "html.parser")
        grid.append(t)
    return grid


def cover_html() -> str:
    members = "".join(f"<tr><td>{n}</td><td>{r}</td></tr>" for n, r in MEMBERS)
    return f"""
<section class="cover">
  <div class="band"></div>
  <div class="inner">
    <div class="kicker">Machine Learning Project · Assignment 03</div>
    <h1>{TITLE}</h1>
    <div class="sub">{SUBTITLE}</div>
    <div class="doc">Project Report</div>
  </div>
  <div class="meta">
    <h3>Group Members</h3>
    <table class="members"><thead><tr><th>Name</th><th>Roll No.</th></tr></thead><tbody>{members}</tbody></table>
    <h3>Project Details</h3>
    <dl>
      <dt>Dataset</dt><dd>Hugging Face <code>jrtec/Superheroes</code> — <code>train.csv</code> and <code>test.csv</code></dd>
      <dt>Inputs</dt><dd>Six ability scores: intelligence, strength, speed, durability, power, combat</dd>
      <dt>Tasks</dt><dd>Data Preparation · Classification · Clustering · Neural Networks</dd>
      <dt>Submission deadline</dt><dd>{DEADLINE}</dd>
    </dl>
  </div>
  <div class="foot">All results come from <code>superhero_project.ipynb</code>, run top to bottom with <code>random_state=42</code> and <code>tf.keras.utils.set_random_seed(42)</code>.</div>
</section>"""


def contents_html(toc, figs, tabs, page_map) -> str:
    def pg(anchor):
        return str(page_map.get(anchor, "")) if page_map else "00"

    def row(cls, anchor, label):
        return (f'<li class="{cls}"><a href="#{anchor}">{label}</a><span class="dots"></span>'
                f'<a class="pg" href="#{anchor}">{pg(anchor)}</a></li>')

    toc_items = "".join(row(c, a, t) for c, a, t in toc)
    fig_items = "".join(row("lf", a, f"<b>{g}</b>{t}") for g, a, t in figs)
    tab_items = "".join(row("lt", a, f"<b>{t_id}</b>{t}") for t_id, a, t in tabs)
    return f"""
<section class="front">
  <h2 id="contents" style="break-before:auto">Contents</h2>
  <ul class="toc">{toc_items}</ul>
</section>
<section class="front">
  <h2 id="lists" style="break-before:auto">List of Figures and Tables</h2>
  <div class="lists">
    <div><h3 style="margin-top:0">Figures</h3><ul class="toc">{fig_items}</ul></div>
    <div><h3 style="margin-top:0">Tables</h3><ul class="toc">{tab_items}</ul></div>
  </div>
</section>"""


def render(html_body: str) -> None:
    html = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{TITLE} — Project Report</title><style>{CSS}</style></head><body>{html_body}</body></html>")
    BUILD_HTML.write_text(html, encoding="utf-8")
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if browser is None:
        sys.exit("Microsoft Edge or Google Chrome is required to print the PDF.")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF_FILE}", BUILD_HTML.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def page_numbers() -> dict[str, int]:
    """Map each anchor to its page by following the links on the contents pages."""
    reader = PdfReader(str(PDF_FILE))
    dests = {name.lstrip("/"): reader.get_destination_page_number(d) + 1 for name, d in reader.named_destinations.items()}
    found = {}
    for page in reader.pages[:4]:
        for annot in page.get("/Annots") or []:
            a = annot.get_object()
            target = a.get("/Dest") or (a.get("/A") or {}).get("/D")
            if target is None:
                continue
            if isinstance(target, list):
                found_page = reader.get_destination_page_number(reader._build_destination("x", target)) + 1  # noqa: SLF001
                name = None
            else:
                name = str(target).lstrip("/"); found_page = dests.get(name)
            if name and found_page:
                found[name] = found_page
    return found


def main() -> None:
    md_text = MD_FILE.read_text(encoding="utf-8")
    render(build_body(md_text, None))
    pages = page_numbers()
    if not pages:
        sys.exit("Could not read link destinations from the first render.")
    render(build_body(md_text, pages))
    BUILD_HTML.unlink(missing_ok=True)

    reader = PdfReader(str(PDF_FILE))
    writer = PdfWriter(clone_from=reader)
    writer.add_metadata({
        "/Title": f"{TITLE} — Project Report",
        "/Author": "; ".join(f"{n} ({r})" for n, r in MEMBERS),
        "/Subject": "Machine Learning Project · Assignment 03",
        "/Keywords": "classification, clustering, neural networks, superheroes",
    })
    # Bookmarks (PDF outline) for every section and subsection.
    md_html = BeautifulSoup(build_body(md_text, pages), "html.parser")
    parent = None
    for h in md_html.select("section ~ h2, h2, h3"):
        if h.get("id") in ("contents", "lists") or h.find_parent(class_="front") or h.find_parent(class_="cover"):
            continue
        anchor = h.get("id")
        if anchor not in pages:
            continue
        label = clean_text(h)
        label = re.sub(r"^\d+ marks\s*", "", label)
        if h.name == "h2":
            parent = writer.add_outline_item(title=label, page_number=pages[anchor] - 1)
        else:
            writer.add_outline_item(title=label, page_number=pages[anchor] - 1, parent=parent)
    writer.page_mode = "/UseOutlines"
    with open(PDF_FILE, "wb") as fh:
        writer.write(fh)
    print(f"Wrote {PDF_FILE.relative_to(ROOT)} — {len(reader.pages)} pages, {len(pages)} anchors resolved.")


if __name__ == "__main__":
    main()
