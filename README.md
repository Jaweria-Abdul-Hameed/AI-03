# Superhero Intelligence System — Assignment 03

*Can six ability scores tell a hero from a villain?*

| Group Member | Roll No. |
| :--- | :--- |
| Mohsin Khan | 24i-3135 |
| Jaweria Abdul Hameed | 24i-3025 |

## Project layout

```text
AI-03/
├── README.md
├── requirements.txt              # pinned library versions (Python 3.12)
├── my_project/                   # ← the submission folder
│   ├── superhero_project.ipynb   # full pipeline, pre-executed (Sections 2–5)
│   ├── data/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── figures/                  # G1–G16 (G12 is the hand-drawn architecture diagram)
│   └── report/
│       ├── Superhero_Intelligence_System_Report.pdf   # final report (T1–T11, G1–G16, Q1–Q12)
│       └── Superhero_Intelligence_System_Report.md    # report source
├── docs/                         # brief and preparation material (not part of the submission)
│   ├── assignment_03.md
│   ├── Superhero_Intelligence_System_Implementation_Checklist.md / .pdf
│   └── Superhero_Intelligence_System_Demo_Prep_Guide.html / .pdf
└── tools/
    └── build_report_pdf.py       # rebuilds the report PDF from the Markdown source
```

## Setup (Windows)

TensorFlow 2.21 needs Python 3.12, so the environment is created with `uv`:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Run the notebook

Run it from inside `my_project/` so the relative paths `data/train.csv` and `data/test.csv` resolve.

```powershell
cd my_project
..\.venv\Scripts\jupyter lab superhero_project.ipynb
```

Or execute it headless, top to bottom:

```powershell
cd my_project
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace superhero_project.ipynb
```

A full run takes about 70 seconds on CPU and rewrites `figures/G1–G16.png` (except G12).

## Rebuild the report PDF

```powershell
uv run --no-project --python 3.12 --with markdown --with beautifulsoup4 --with pypdf python tools\build_report_pdf.py
```

The script turns `my_project/report/Superhero_Intelligence_System_Report.md` into a formatted A4 PDF with a cover page, contents and lists of figures and tables (with page numbers), running headers and footers, and PDF bookmarks. It prints with Microsoft Edge (or Chrome) in headless mode. Its helper libraries run in a temporary `uv` environment, so the project `.venv` stays exactly as pinned in `requirements.txt`.
