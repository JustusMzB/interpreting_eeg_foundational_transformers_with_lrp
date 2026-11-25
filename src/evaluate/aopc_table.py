"""Builds the AOPC results table (paper Table A5) from one run's AOPC results and writes CSV + booktabs LaTeX."""
import os
from typing import Optional

import numpy as np
import pandas as pd

# Result keys written by script/run_aopc_eval_suite.py -> row labels, in table order.
METHODS = {
    "morf": "Per-sample LRP",
    "gxi_morf": "Per-sample G×I",
    "global": "Global aggregate LRP",
    "gxi_global": "Global aggregate G×I",
    "rand": "Random baseline",
}
RANDOM_KEY = "rand"


def _latex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(a, b)
    return text


def build_aopc_table(result: dict) -> pd.DataFrame:
    """Summarises per-sample AOPC values of one run.

    Columns: n, mean, sd (across samples), rel_random (mean / mean of the random baseline).
    """
    per_sample = result["per_sample_aopc"]
    values = {k: np.asarray(per_sample[k], dtype=float) for k in METHODS if k in per_sample}
    if RANDOM_KEY not in values:
        raise KeyError("Results contain no random baseline ('rand'); cannot compute 'relative to random'.")
    n = len(values[RANDOM_KEY])
    if any(len(v) != n for v in values.values()):
        raise ValueError("All methods must be evaluated on the same samples.")

    rand_mean = values[RANDOM_KEY].mean()

    rows = []
    for key, label in METHODS.items():
        if key not in values:
            continue
        v = values[key]
        rows.append({
            "method": label,
            "n": n,
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "rel_random": v.mean() / rand_mean,
        })
    return pd.DataFrame(rows).set_index("method")


def table_to_latex(table: pd.DataFrame, caption: str, label: str = "tab:aopc") -> str:
    """Booktabs LaTeX table (requires \\usepackage{booktabs})."""
    def esc(s: str) -> str:
        return _latex_escape(s).replace("×", r"$\times$")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Attribution method & AOPC$_{\text{adapted}}$, mean (SD) & Relative to random \\",
        r"\midrule",
    ]
    for method, r in table.iterrows():
        if method == METHODS[RANDOM_KEY]:
            lines.append(r"\midrule")
            rel = "1.00"
        else:
            rel = f"{r.rel_random:.2f}"
        lines.append(f"{esc(method)} & {r['mean']:.3f} ({r.sd:.3f}) & {rel} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_aopc_table(result: dict, out_prefix: str, caption: Optional[str] = None) -> pd.DataFrame:
    """Writes `<out_prefix>.csv` and `<out_prefix>.tex` and returns the table."""
    table = build_aopc_table(result)
    n = int(table["n"].iloc[0])
    steps = len(result.get("morf_curve", [])) or "L"
    if caption is None:
        # Orderings are described in the paper text (App. E.1) and in aopc_evaluation_methodology.txt.
        run = _latex_escape(str(result.get("run", "")).replace("run_", "run "))
        caption = (f"Adapted AOPC (Eq.~2), {steps} steps, {run} {_latex_escape(result.get('split', 'test'))} split "
                   f"($n={n}$). Mean (SD) and ratio to random.")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    table.to_csv(f"{out_prefix}.csv", float_format="%.6f")
    with open(f"{out_prefix}.tex", "w") as f:
        f.write(table_to_latex(table, caption))
    return table
