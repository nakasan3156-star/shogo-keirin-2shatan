#!/usr/bin/env python3
"""Faster launcher with identical statistics/model rules.

Reuses only computations that are mathematically identical between the no-venue and
venue variants: component predictions, battle maps, and neutral scenario rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import run_venue_edge_analysis as safe  # installs safe empty-cell handling

v = safe.v

_scenario_cache = {}
_shared = {}


def cached_scenario_rows(riders: pd.DataFrame, battle_map: dict, venue_table: pd.DataFrame) -> pd.DataFrame:
    ids = tuple(riders.race_id.astype(str).unique().tolist())
    key = ids
    if key not in _scenario_cache:
        neutral = pd.DataFrame({"venue_code": riders.venue_code.astype(str).unique()})
        for c in ["back_win_adj", "bandte_win_adj", "third_top3_adj", "makuri_win_adj"]:
            neutral[c] = 0.0
        base_rows = v.base.scenario_pair_rows(riders, battle_map, neutral)
        race_venue = riders.groupby("race_id").venue_code.first().astype(str).to_dict()
        base_rows["venue_code"] = base_rows.race_id.map(race_venue).astype(str)
        _scenario_cache[key] = base_rows
    s = _scenario_cache[key].copy()
    if s.empty:
        return s
    vt = venue_table.copy()
    vt["venue_code"] = vt.venue_code.astype(str)
    idx = vt.set_index("venue_code")

    def getadj(metric):
        col = metric + "_adj"
        if col not in idx:
            return np.zeros(len(s))
        return s.venue_code.map(idx[col]).fillna(0).to_numpy(float)

    back_win = getadj("back_win")
    back_top2 = getadj("back_top2")
    bandte_win = getadj("bandte_win")
    bandte_top2 = getadj("bandte_top2")
    third = getadj("third_top3")
    cross = getadj("otherwin_backline_second")
    mak = getadj("win_makuri")
    same = getadj("same_line_top2")
    s["v_first_back_win"] = s.first_is_back.to_numpy() * back_win
    s["v_second_back_top2"] = s.second_is_back.to_numpy() * back_top2
    s["v_first_bandte_win"] = s.first_is_bandte_of_back.to_numpy() * bandte_win
    s["v_second_bandte_top2"] = s.second_is_bandte_of_back.to_numpy() * bandte_top2
    s["v_second_third_top3"] = s.second_is_third_of_back.to_numpy() * third
    s["v_first_makuri"] = s.first_other_leader.to_numpy() * mak
    second_backline = np.maximum.reduce([
        s.second_is_back.to_numpy(),
        s.second_is_bandte_of_back.to_numpy(),
        s.second_is_third_of_back.to_numpy(),
    ])
    s["v_crossline_backline_second"] = s.first_other_leader.to_numpy() * second_backline * cross
    s["v_same_line_top2"] = s.same_line.to_numpy() * same
    return s


def fast_run_pair_variant(name, d, races25, results25, odds25, races26, results26, odds26, venue_table):
    if not _shared:
        component = d[d.race_date.le(v.DISC_END)].copy()
        pair = d[(d.race_date.gt(v.DISC_END)) & (d.race_date.le(v.PAIR_END))].copy()
        valid = d[(d.race_date.gt(v.PAIR_END)) & (d.race_date.le(v.VALID_END))].copy()
        test = d[d.race_date.between(v.TEST_START, v.TEST_END)].copy()
        ext = d[d.race_date.between(v.EXT_START, v.EXT_END)].copy()
        cats, _ = v.base.category_effects(component)
        v.base.add_component_predictions([component, pair, valid, test, ext], component, cats)
        events = v.base.race_event_table(d)
        train_ev = events[events.race_date.le(v.DISC_END)]
        em = LogisticRegression(max_iter=500, class_weight="balanced", random_state=v.SEED).fit(
            train_ev[v.base.EVENT_FEATURES].fillna(0), train_ev.early_battle_label
        )
        maps = {}
        for nm, f in [("pair", pair), ("valid", valid), ("test", test), ("ext", ext)]:
            e = events[events.race_id.isin(f.race_id.unique())].copy()
            maps[nm] = dict(zip(e.race_id, em.predict_proba(e[v.base.EVENT_FEATURES].fillna(0))[:, 1]))
        _shared.update(component=component, pair=pair, valid=valid, test=test, ext=ext, maps=maps)

    pair = _shared["pair"]
    valid = _shared["valid"]
    test = _shared["test"]
    ext = _shared["ext"]
    maps = _shared["maps"]

    pm = v.fit_pair(cached_scenario_rows(pair, maps["pair"], venue_table), results25)
    pv = v.predict_pairs(cached_scenario_rows(valid, maps["valid"], venue_table), pm)
    pt = v.predict_pairs(cached_scenario_rows(test, maps["test"], venue_table), pm)
    pe = v.predict_pairs(cached_scenario_rows(ext, maps["ext"], venue_table), pm)
    vm = v.base.attach_market(pv, odds25, races25, results25)
    tm = v.base.attach_market(pt, odds25, races25, results25)
    xm = v.base.attach_market(pe, odds26, races26, results26)
    v.calibrate(vm, [tm, xm])
    _, _, joint = v.base.reliability_tables(vm)
    tm = v.base.apply_reliability(tm, joint)
    xm = v.base.apply_reliability(xm, joint)
    _, tb = v.base.portfolio(tm, **v.FROZEN_RULE)
    _, xb = v.base.portfolio(xm, **v.FROZEN_RULE)
    return {
        "name": name,
        "test_market": tm,
        "ext_market": xm,
        "test_bets": tb,
        "ext_bets": xb,
        "test": v.path_metrics(tb, test.race_id.nunique()),
        "external": v.path_metrics(xb, ext.race_id.nunique()),
    }


v.scenario_rows = cached_scenario_rows
v.run_pair_variant = fast_run_pair_variant

if __name__ == "__main__":
    v.main()
