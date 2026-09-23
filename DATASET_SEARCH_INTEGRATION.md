# Dataset Search integration

Integrated into `app/ui/main_window.py` as a new **Dataset Search** navigation item.

## Features

- Real-time public dataset search via Hugging Face Hub
- Search by keyword
- Dataset metadata cards
- Supported tabular file discovery: CSV, TSV, JSON, JSONL, Parquet, XLS/XLSX
- Open source page in browser
- Direct file download to `data/downloads/`
- `Download & Analyze` for supported pandas-readable formats
- Search/download runs in background Qt threads so the UI remains responsive
- Existing UI appearance and existing analysis/research backends are preserved

## Files added

- `app/dataset_search/__init__.py`
- `app/dataset_search/huggingface.py`
- `app/dataset_search/README.md`
- `app/ui/dataset_search_page.py`

## Files updated

- `app/ui/main_window.py`

No new third-party dependency is required for the search/download feature beyond the existing `httpx` and `pandas` already used by the project.
