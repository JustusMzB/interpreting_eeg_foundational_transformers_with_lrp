"""Electrode relevance statistics for arousal prediction (Mann–Whitney U vs. background, Spearman vs. balanced accuracy).

The analysis is implemented only in src/notebooks/quantifying_attr_on_arousal.ipynb. This script executes that
notebook in place, so the results stay documented next to the code that produced them.
Outputs: the executed notebook and results/apa_mwu/arousal_* (booktabs tables, statistics cache, CSVs, topomaps).

Usage (from any directory): python script/quantify_attr_arousal.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path(__file__).resolve().parents[1] / "src" / "notebooks" / "quantifying_attr_on_arousal.ipynb"


def main():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    # run with the notebook's folder as working directory: its relative paths (../../out, ../../src) start there
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    nbformat.write(nb, NOTEBOOK)
    print(f"executed {NOTEBOOK}; outputs in results/apa_mwu/")


if __name__ == "__main__":
    main()
