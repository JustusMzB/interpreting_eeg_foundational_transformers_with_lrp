import numpy as np
import pandas as pd
import pytest

from evaluate.aopc_table import build_aopc_table, write_aopc_table


def _result(n=50):
    rng = np.random.default_rng(0)
    return {"run": "run_3", "split": "test", "morf_curve": [0] * 10,
            "per_sample_ordering": "prediction", "global_ordering": "magnitude",
            "per_sample_aopc": {"morf": (0.8 + rng.normal(0, .1, n)).tolist(),
                                "gxi_morf": (0.4 + rng.normal(0, .1, n)).tolist(),
                                "global": (0.3 + rng.normal(0, .1, n)).tolist(),
                                "gxi_global": (0.2 + rng.normal(0, .1, n)).tolist(),
                                "rand": (0.1 + rng.normal(0, .01, n)).tolist()}}


def test_table_values_and_relative_to_random():
    res = _result()
    t = build_aopc_table(res)
    v = np.asarray(res["per_sample_aopc"]["morf"])
    assert t.loc["Per-sample LRP", "mean"] == pytest.approx(v.mean())
    assert t.loc["Per-sample LRP", "sd"] == pytest.approx(v.std(ddof=1))
    rand = np.mean(res["per_sample_aopc"]["rand"])
    assert t.loc["Per-sample LRP", "rel_random"] == pytest.approx(v.mean() / rand)
    assert t.loc["Random baseline", "rel_random"] == pytest.approx(1.0)
    assert list(t.index)[-1] == "Random baseline"


def test_writes_csv_and_booktabs_latex(tmp_path):
    t = write_aopc_table(_result(), str(tmp_path / "aopc_table_run_3"))
    csv = pd.read_csv(tmp_path / "aopc_table_run_3.csv", index_col=0)
    assert csv.loc["Per-sample LRP", "rel_random"] == pytest.approx(t.loc["Per-sample LRP", "rel_random"])
    tex = (tmp_path / "aopc_table_run_3.tex").read_text()
    for token in (r"\toprule", r"\midrule", r"\bottomrule", "Relative to random", r"$\times$"):
        assert token in tex
    assert "×" not in tex  # unicode replaced for LaTeX


def test_caption_is_latex_safe(tmp_path):
    write_aopc_table(_result(), str(tmp_path / "t"))
    tex = (tmp_path / "t.tex").read_text()
    caption = [l for l in tex.splitlines() if l.startswith(r"\caption")][0]
    assert "run_3" not in caption and "run 3" in caption
    assert "_" not in caption.replace(r"\_", "")


def test_no_confidence_intervals(tmp_path):
    t = write_aopc_table(_result(), str(tmp_path / "t"))
    assert list(t.columns) == ["n", "mean", "sd", "rel_random"]
    tex = (tmp_path / "t.tex").read_text()
    assert "CI" not in tex and "bootstrap" not in tex
