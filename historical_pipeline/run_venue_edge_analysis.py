#!/usr/bin/env python3
"""Safe launcher for venue_edge_analysis.

Only fixes the representation of empty venue×period cells: missing sample counts are
interpreted as n=0. Statistical rules, thresholds, features, and time splits are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import venue_edge_analysis as v


def _num(x, default=0.0):
    try:
        return float(x) if pd.notna(x) else float(default)
    except (TypeError, ValueError):
        return float(default)


def safe_build_adjustment_table(resid: pd.DataFrame):
    piv = resid.pivot_table(
        index=["venue_code", "venue_name", "metric"],
        columns="split",
        values=["n", "actual_rate", "expected_rate", "residual_pp"],
        aggfunc="first",
    ).reset_index()
    piv.columns = [
        "_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else c
        for c in piv.columns
    ]

    def col(base, split):
        return f"{base}_{split}"

    rows = []
    overall_disc = {}
    for m in v.METRICS:
        z = resid[(resid.metric == m) & (resid.split == "discovery_2025_01_06")]
        overall_disc[m] = float(np.average(z.actual_rate, weights=z.n)) if len(z) else 0.5

    for _, r in piv.iterrows():
        metric = r["metric"]
        nd = _num(r.get(col("n", "discovery_2025_01_06"), 0), 0)
        nv = _num(r.get(col("n", "validation_2025_09_10"), 0), 0)
        ad = _num(r.get(col("actual_rate", "discovery_2025_01_06"), np.nan), np.nan)
        ed = _num(r.get(col("expected_rate", "discovery_2025_01_06"), np.nan), np.nan)
        rd = _num(r.get(col("residual_pp", "discovery_2025_01_06"), np.nan), np.nan)
        rv = _num(r.get(col("residual_pp", "validation_2025_09_10"), np.nan), np.nan)

        enough = nd >= v.MIN_DISC_N and nv >= v.MIN_VALID_N
        same = np.isfinite(rd) and np.isfinite(rv) and rd * rv > 0
        material = np.isfinite(rd) and np.isfinite(rv) and abs(rd) >= 0.75 and abs(rv) >= 0.50
        accepted = bool(enough and same and material)

        prior = overall_disc.get(metric, 0.5)
        if np.isfinite(ad) and np.isfinite(ed) and nd > 0:
            obs = (ad * nd + v.SHRINK_K * prior) / (nd + v.SHRINK_K)
            exp = (ed * nd + v.SHRINK_K * prior) / (nd + v.SHRINK_K)
            adj = float(np.clip(v.logit(obs) - v.logit(exp), -v.MAX_ADJ, v.MAX_ADJ))
        else:
            adj = 0.0
        if not accepted:
            adj = 0.0

        rows.append({
            "venue_code": r["venue_code"],
            "venue_name": r["venue_name"],
            "metric": metric,
            "metric_ja": v.METRIC_JA[metric],
            "discovery_n": int(nd),
            "validation_n": int(nv),
            "discovery_residual_pp": rd,
            "validation_residual_pp": rv,
            "candidate_reproduced": accepted,
            "logit_adjustment": adj,
        })

    effects = pd.DataFrame(rows)
    use_metrics = {
        "back_win", "back_top2", "bandte_win", "bandte_top2", "third_top3",
        "otherwin_backline_second", "win_makuri", "same_line_top2",
    }
    venue_rows = []
    for (vc, vn), g in effects.groupby(["venue_code", "venue_name"]):
        rec = {"venue_code": vc, "venue_name": vn}
        for m in use_metrics:
            z = g[g.metric.eq(m)]
            rec[m + "_adj"] = float(z.logit_adjustment.iloc[0]) if len(z) else 0.0
            rec[m + "_candidate"] = bool(z.candidate_reproduced.iloc[0]) if len(z) else False
        venue_rows.append(rec)
    return effects, pd.DataFrame(venue_rows)


v.build_adjustment_table = safe_build_adjustment_table

if __name__ == "__main__":
    v.main()
