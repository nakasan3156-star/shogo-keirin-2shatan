#!/usr/bin/env python3
"""全国競輪場の再現性ある場差を検証し、2車単ROIで採否を決める。

時系列を厳格分離する。
- 場差発見・能力モデル: 2025-01..06
- 2車単モデル:        2025-07..08
- 場差再現確認/校正:   2025-09..10
- ROIゲート:           2025-11..12
- 最終未使用検証:      2026-01..02

2026-01..02 の結果は補正値・閾値・採用候補の調整に使用しない。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import chatgpt_baseline_backtest as base

SEED = 20260810
DISC_END = 20250630
PAIR_END = 20250831
VALID_END = 20251031
TEST_START = 20251101
TEST_END = 20251231
EXT_START = 20260101
EXT_END = 20260228
FROZEN_RULE = {"max_points": 5, "min_ev": 2.0, "min_prob": 0.03, "confidence_max": 1.0}

MIN_DISC_N = 50
MIN_VALID_N = 25
MIN_TEST_N = 15
MIN_EXT_N = 15
SHRINK_K = 100.0
MAX_ADJ = 0.45

RIDER_MODEL_FEATURES = [
    "score", "escape", "makuri", "sashi", "mark", "b_count", "h_count", "s_count",
    "finish_1", "finish_2", "finish_3", "finish_out", "win_rate", "top2_rate", "top3_rate",
    "actual_starters", "line_count", "line_position", "line_size", "rank_level2", "style_code2",
    "day_no", "distance_m", "wind_speed", "score_rel", "b_rel", "h_rel", "escape_rel",
    "makuri_rel", "sashi_rel", "mark_rel", "score_rank", "b_rank", "h_rank", "escape_rank",
    "makuri_rank", "b_top_gap", "h_top_gap", "score_top_gap", "line_score_rel", "line_b_rel",
    "line_h_rel", "is_leader", "is_bandte", "is_third", "is_single", "is_self_power",
    "two_line", "three_line", "fragmented", "leader_b_gap", "leader_h_gap", "escape_leader_count",
]

RACE_MODEL_FEATURES = [
    "actual_starters", "line_count", "day_no", "distance_m", "wind_speed", "class_code",
    "b_top_gap", "h_top_gap", "score_top_gap", "leader_b_gap", "leader_h_gap", "escape_leader_count",
    "max_leader_b", "max_leader_h", "mean_leader_b", "mean_leader_h", "max_score", "score_sd",
    "max_line_size", "min_line_size", "three_plus_line_count", "two_line", "three_line", "fragmented",
]

METRICS = [
    "back_win", "back_second", "back_third", "back_top2",
    "bandte_win", "bandte_top2", "bandte_top3", "third_top3",
    "otherwin_backline_second", "front_selfpower_second",
    "win_escape", "win_makuri", "win_sashi", "same_line_top2", "three_man_sweep",
]

METRIC_JA = {
    "back_win": "B取得者1着率",
    "back_second": "B取得者2着率",
    "back_third": "B取得者3着率",
    "back_top2": "B取得者2連対率",
    "bandte_win": "B取得ライン番手1着率",
    "bandte_top2": "B取得ライン番手2連対率",
    "bandte_top3": "B取得ライン番手3連対率",
    "third_top3": "B取得ライン3番手3着内率",
    "otherwin_backline_second": "別線1着時B取得ライン2着残り率",
    "front_selfpower_second": "前で踏んだ自力選手2着残り率",
    "win_escape": "逃げ決着率",
    "win_makuri": "捲り決着率",
    "win_sashi": "差し決着率",
    "same_line_top2": "1-2着同ライン率",
    "three_man_sweep": "3車ライン上位3着独占率",
}

PAIR_VENUE_FEATURES = [
    "v_first_back_win", "v_second_back_top2",
    "v_first_bandte_win", "v_second_bandte_top2",
    "v_second_third_top3", "v_first_makuri",
    "v_crossline_backline_second", "v_same_line_top2",
]
PAIR_FEATURES_PLUS = base.PAIR_FEATURES + PAIR_VENUE_FEATURES


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def load(root: Path, label: str):
    races = pd.read_csv(root / f"races_{label}.csv", dtype={"race_id": str, "venue_code": str})
    riders = pd.read_csv(root / f"rider_features_{label}.csv", dtype={"race_id": str, "venue_code": str})
    results = pd.read_csv(root / f"official_results_{label}.csv", dtype={"race_id": str})
    odds = pd.read_csv(root / f"exacta_odds_{label}.csv", dtype={"race_id": str})
    for d in [races, riders, results, odds]:
        for c in [
            "race_date", "car_no", "first_car", "second_car", "finish_order", "actual_back",
            "actual_start", "exacta_odds", "line_no", "line_position", "line_size", "actual_starters",
            "line_count", "day_no", "distance_m", "wind_speed",
        ]:
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
    return races, riders, results, odds


def prepare_rider_frame(races25, riders25, results25, races26, riders26, results26):
    races = pd.concat([races25, races26], ignore_index=True, sort=False)
    riders = pd.concat([riders25, riders26], ignore_index=True, sort=False)
    results = pd.concat([results25, results26], ignore_index=True, sort=False)
    d = base.add_prerace_features(riders, races, results)
    lab = results[[
        "race_id", "car_no", "finish_order", "actual_back", "actual_start", "winning_move",
        "final_lap_time", "result_comment"
    ]].copy()
    d = d.merge(lab, on=["race_id", "car_no"], how="inner")
    d = d[pd.to_numeric(d.finish_order, errors="coerce").gt(0)].copy()
    for c in ["race_date", "car_no", "finish_order", "actual_back", "actual_start", "line_no", "line_position", "line_size", "actual_starters", "line_count", "day_no", "distance_m", "wind_speed"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["rank_level2"] = d["rank"].map({"A3": 1, "A2": 2, "A1": 3, "S2": 4, "S1": 5, "SS": 6}).fillna(0)
    d["style_code2"] = d["style"].map({"逃": 3, "両": 2, "追": 1}).fillna(0)
    d["class_group"] = np.where(d["rank"].astype(str).str.startswith("S"), "S級", "A級")
    d["class_code"] = np.where(d["class_group"].eq("S級"), 2, 1)
    d["starter_group"] = np.where(d.actual_starters.eq(9), "9車", np.where(d.actual_starters.eq(7), "7車", d.actual_starters.fillna(0).astype(int).astype(str)+"車"))
    d["formation"] = np.select([d.line_count.eq(2), d.line_count.eq(3), d.line_count.ge(4)], ["二分戦", "三分戦", "細切れ"], default="その他")
    d["y_win"] = d.finish_order.eq(1).astype(int)
    d["y_top2"] = d.finish_order.le(2).astype(int)
    d["y_top3"] = d.finish_order.le(3).astype(int)
    d["y_back"] = d.actual_back.fillna(0).astype(int)

    back = d.loc[d.actual_back.eq(1), ["race_id", "car_no", "line_no"]].copy()
    back = back.sort_values(["race_id", "car_no"]).drop_duplicates("race_id").rename(columns={"car_no":"back_car", "line_no":"back_line"})
    d = d.merge(back, on="race_id", how="left")
    d["is_back_holder"] = d.car_no.eq(d.back_car)
    d["is_back_bandte"] = d.line_no.eq(d.back_line) & d.line_position.eq(2)
    d["is_back_third"] = d.line_no.eq(d.back_line) & d.line_position.eq(3)
    d["is_other_leader"] = d.line_no.ne(d.back_line) & d.line_position.eq(1)
    d["is_front_selfpower"] = d.is_leader.eq(1) & d.is_self_power.eq(1) & (d.actual_start.eq(1) | d.actual_back.eq(1))
    return d


def make_race_frame(d: pd.DataFrame) -> pd.DataFrame:
    leaders = d[d.is_leader.eq(1)].copy()
    agg = d.groupby("race_id", sort=False).agg(
        race_date=("race_date","first"), venue_code=("venue_code","first"), venue_name=("venue_name","first"),
        actual_starters=("actual_starters","first"), line_count=("line_count","first"), day_no=("day_no","first"),
        distance_m=("distance_m","first"), wind_speed=("wind_speed","first"), class_code=("class_code","first"),
        class_group=("class_group","first"), starter_group=("starter_group","first"), formation=("formation","first"),
        b_top_gap=("b_top_gap","first"), h_top_gap=("h_top_gap","first"), score_top_gap=("score_top_gap","first"),
        leader_b_gap=("leader_b_gap","first"), leader_h_gap=("leader_h_gap","first"), escape_leader_count=("escape_leader_count","first"),
        max_score=("score","max"), score_sd=("score","std"), max_line_size=("line_size","max"), min_line_size=("line_size","min"),
    ).reset_index()
    l = leaders.groupby("race_id").agg(max_leader_b=("b_count","max"), max_leader_h=("h_count","max"), mean_leader_b=("b_count","mean"), mean_leader_h=("h_count","mean"), three_plus_line_count=("line_size", lambda x: int((x>=3).sum()))).reset_index()
    agg = agg.merge(l, on="race_id", how="left")
    agg["two_line"] = agg.line_count.eq(2).astype(int)
    agg["three_line"] = agg.line_count.eq(3).astype(int)
    agg["fragmented"] = agg.line_count.ge(4).astype(int)

    winners = d[d.finish_order.eq(1)][["race_id","line_no","car_no","winning_move"]].copy().rename(columns={"line_no":"winner_line","car_no":"winner_car","winning_move":"winner_move"})
    seconds = d[d.finish_order.eq(2)][["race_id","line_no","car_no"]].copy().rename(columns={"line_no":"second_line","car_no":"second_car"})
    thirds = d[d.finish_order.eq(3)][["race_id","line_no","car_no"]].copy().rename(columns={"line_no":"third_line","car_no":"third_car"})
    back = d[d.is_back_holder][["race_id","back_car","back_line","score","b_count","h_count","line_size"]].copy().rename(columns={"score":"back_score","b_count":"back_b_count","h_count":"back_h_count","line_size":"back_line_size"})
    back = back.drop_duplicates("race_id")
    line_sizes = d[["race_id","line_no","line_size"]].drop_duplicates(["race_id","line_no"]).rename(columns={"line_no":"winner_line","line_size":"winner_line_size"})
    r = agg.merge(winners, on="race_id", how="left").merge(seconds, on="race_id", how="left").merge(thirds, on="race_id", how="left").merge(back, on="race_id", how="left").merge(line_sizes,on=["race_id","winner_line"],how="left")
    r["otherwin"] = r.winner_line.ne(r.back_line)
    r["otherwin_backline_second"] = r.otherwin & r.second_line.eq(r.back_line)
    r["same_line_top2"] = r.winner_line.eq(r.second_line)
    r["three_man_sweep"] = r.winner_line.eq(r.second_line) & r.winner_line.eq(r.third_line) & r.winner_line_size.ge(3)
    r["win_escape"] = r.winner_move.eq("逃")
    r["win_makuri"] = r.winner_move.eq("捲")
    r["win_sashi"] = r.winner_move.eq("差")
    return r


def metric_frame(metric: str, d: pd.DataFrame, races: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if metric == "back_win":
        x=d[d.is_back_holder].copy(); x["y"]=x.finish_order.eq(1).astype(int); return x,RIDER_MODEL_FEATURES
    if metric == "back_second":
        x=d[d.is_back_holder].copy(); x["y"]=x.finish_order.eq(2).astype(int); return x,RIDER_MODEL_FEATURES
    if metric == "back_third":
        x=d[d.is_back_holder].copy(); x["y"]=x.finish_order.eq(3).astype(int); return x,RIDER_MODEL_FEATURES
    if metric == "back_top2":
        x=d[d.is_back_holder].copy(); x["y"]=x.finish_order.le(2).astype(int); return x,RIDER_MODEL_FEATURES
    if metric == "bandte_win":
        x=d[d.is_back_bandte].copy(); x["y"]=x.finish_order.eq(1).astype(int); return x,RIDER_MODEL_FEATURES
    if metric == "bandte_top2":
        x=d[d.is_back_bandte].copy(); x["y"]=x.finish_order.le(2).astype(int); return x,RIDER_MODEL_FEATURES
    if metric == "bandte_top3":
        x=d[d.is_back_bandte].copy(); x["y"]=x.finish_order.le(3).astype(int); return x,RIDER_MODEL_FEATURES
    if metric == "third_top3":
        x=d[d.is_back_third].copy(); x["y"]=x.finish_order.le(3).astype(int); return x,RIDER_MODEL_FEATURES
    if metric == "front_selfpower_second":
        x=d[d.is_front_selfpower].copy(); x["y"]=x.finish_order.eq(2).astype(int); return x,RIDER_MODEL_FEATURES
    if metric == "otherwin_backline_second":
        x=races[races.otherwin].copy(); x["y"]=x.otherwin_backline_second.astype(int); return x,RACE_MODEL_FEATURES
    if metric in {"win_escape","win_makuri","win_sashi","same_line_top2","three_man_sweep"}:
        x=races.copy(); x["y"]=x[metric].astype(int); return x,RACE_MODEL_FEATURES
    raise KeyError(metric)


def fit_hgb(train: pd.DataFrame, features: list[str]):
    model = HistGradientBoostingClassifier(learning_rate=.045, max_iter=160, max_leaf_nodes=19, min_samples_leaf=45, l2_regularization=3.0, random_state=SEED)
    model.fit(train[features].fillna(-99), train.y.astype(int))
    return model


def discovery_oof(train: pd.DataFrame, features: list[str]) -> np.ndarray:
    if len(train) < 100 or train.y.nunique() < 2:
        return np.full(len(train), float(train.y.mean()) if len(train) else 0.0)
    groups = train.race_id.astype(str)
    unique_groups = groups.nunique()
    folds = min(5, unique_groups)
    if folds < 2:
        return np.full(len(train), float(train.y.mean()))
    pred = np.zeros(len(train), dtype=float)
    gkf = GroupKFold(n_splits=folds)
    X = train[features].fillna(-99)
    y = train.y.astype(int)
    for tr, va in gkf.split(X, y, groups):
        if y.iloc[tr].nunique() < 2:
            pred[va] = y.iloc[tr].mean()
            continue
        m = fit_hgb(train.iloc[tr].copy(), features)
        pred[va] = m.predict_proba(X.iloc[va])[:,1]
    return pred


def split_name(date):
    if date <= DISC_END: return "discovery_2025_01_06"
    if date <= PAIR_END: return "pair_train_2025_07_08"
    if date <= VALID_END: return "validation_2025_09_10"
    if date <= TEST_END: return "locked_test_2025_11_12"
    if EXT_START <= date <= EXT_END: return "external_2026_01_02"
    return "other"


def venue_metric_residuals(d: pd.DataFrame, races: pd.DataFrame):
    all_rows=[]; model_meta=[]
    for metric in METRICS:
        f,features=metric_frame(metric,d,races)
        f=f[f.race_date.between(20250101,EXT_END)].copy().reset_index(drop=True)
        disc=f[f.race_date.le(DISC_END)].copy()
        if disc.empty or disc.y.nunique()<2:
            continue
        disc_pred=discovery_oof(disc,features)
        model=fit_hgb(disc,features)
        f["expected"]=model.predict_proba(f[features].fillna(-99))[:,1]
        if "car_no" in f.columns:
            mp={(r,c):p for r,c,p in zip(disc.race_id,disc.car_no,disc_pred)}
            mask=f.race_date.le(DISC_END)
            f.loc[mask,"expected"]=[mp.get((r,c),e) for r,c,e in zip(f.loc[mask,"race_id"],f.loc[mask,"car_no"],f.loc[mask,"expected"])]
        else:
            mp=dict(zip(disc.race_id,disc_pred));mask=f.race_date.le(DISC_END);f.loc[mask,"expected"]=f.loc[mask,"race_id"].map(mp).fillna(f.loc[mask,"expected"])
        f["split"] = f.race_date.astype(int).map(split_name)
        f["resid"] = f.y.astype(float)-f.expected.astype(float)
        model_meta.append({"metric":metric,"train_n":len(disc),"positive_rate":float(disc.y.mean()),"features":len(features)})
        for (split,vc,vn),g in f.groupby(["split","venue_code","venue_name"],dropna=False):
            if split=="other": continue
            n=len(g); actual=float(g.y.mean()); expected=float(g.expected.mean()); resid=float((g.y-g.expected).mean())
            se=float(g.resid.std(ddof=1)/math.sqrt(n)) if n>1 else None
            all_rows.append({"metric":metric,"metric_ja":METRIC_JA[metric],"split":split,"venue_code":str(vc).zfill(2),"venue_name":vn,"n":n,"actual_rate":actual,"expected_rate":expected,"residual_pp":resid*100,"residual_se_pp":None if se is None else se*100,"ci95_low_pp":None if se is None else (resid-1.96*se)*100,"ci95_high_pp":None if se is None else (resid+1.96*se)*100})
    return pd.DataFrame(all_rows),pd.DataFrame(model_meta)


def build_adjustment_table(resid: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    piv=resid.pivot_table(index=["venue_code","venue_name","metric"],columns="split",values=["n","actual_rate","expected_rate","residual_pp"],aggfunc="first").reset_index()
    piv.columns=["_".join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in piv.columns]
    def col(base,split): return f"{base}_{split}"
    rows=[]
    overall_disc={m:float(resid[(resid.metric==m)&(resid.split=="discovery_2025_01_06")].pipe(lambda x: np.average(x.actual_rate,weights=x.n) if len(x) else 0.5)) for m in METRICS}
    for _,r in piv.iterrows():
        metric=r["metric"]; nd=float(r.get(col("n","discovery_2025_01_06"),0) or 0); nv=float(r.get(col("n","validation_2025_09_10"),0) or 0)
        ad=float(r.get(col("actual_rate","discovery_2025_01_06"),np.nan)); ed=float(r.get(col("expected_rate","discovery_2025_01_06"),np.nan)); rd=float(r.get(col("residual_pp","discovery_2025_01_06"),np.nan)); rv=float(r.get(col("residual_pp","validation_2025_09_10"),np.nan))
        enough=nd>=MIN_DISC_N and nv>=MIN_VALID_N
        same=np.isfinite(rd) and np.isfinite(rv) and rd*rv>0
        material=np.isfinite(rd) and np.isfinite(rv) and abs(rd)>=0.75 and abs(rv)>=0.50
        accepted=bool(enough and same and material)
        prior=overall_disc.get(metric,0.5)
        if np.isfinite(ad) and np.isfinite(ed) and nd>0:
            obs=(ad*nd+SHRINK_K*prior)/(nd+SHRINK_K); exp=(ed*nd+SHRINK_K*prior)/(nd+SHRINK_K)
            adj=float(np.clip(logit(obs)-logit(exp),-MAX_ADJ,MAX_ADJ))
        else: adj=0.0
        if not accepted: adj=0.0
        rows.append({"venue_code":r["venue_code"],"venue_name":r["venue_name"],"metric":metric,"metric_ja":METRIC_JA[metric],"discovery_n":int(nd),"validation_n":int(nv),"discovery_residual_pp":rd,"validation_residual_pp":rv,"candidate_reproduced":accepted,"logit_adjustment":adj})
    effects=pd.DataFrame(rows)
    use_metrics={"back_win","back_top2","bandte_win","bandte_top2","third_top3","otherwin_backline_second","win_makuri","same_line_top2"}
    venue_rows=[]
    for (vc,vn),g in effects.groupby(["venue_code","venue_name"]):
        rec={"venue_code":vc,"venue_name":vn}
        for m in use_metrics:
            z=g[g.metric.eq(m)]
            rec[m+"_adj"]=float(z.logit_adjustment.iloc[0]) if len(z) else 0.0
            rec[m+"_candidate"]=bool(z.candidate_reproduced.iloc[0]) if len(z) else False
        venue_rows.append(rec)
    return effects,pd.DataFrame(venue_rows)


def zero_venue_table(codes_names: pd.DataFrame) -> pd.DataFrame:
    cols=[m+"_adj" for m in ["back_win","back_top2","bandte_win","bandte_top2","third_top3","otherwin_backline_second","win_makuri","same_line_top2"]]
    z=codes_names[["venue_code","venue_name"]].drop_duplicates().copy()
    for c in cols:z[c]=0.0
    return z


def scenario_rows(riders: pd.DataFrame, battle_map: dict, venue_table: pd.DataFrame) -> pd.DataFrame:
    neutral=pd.DataFrame({"venue_code":venue_table.venue_code.astype(str).unique()})
    for c in ["back_win_adj","bandte_win_adj","third_top3_adj","makuri_win_adj"]:
        neutral[c]=0.0
    s=base.scenario_pair_rows(riders,battle_map,neutral)
    if s.empty:return s
    race_venue=riders.groupby("race_id").venue_code.first().astype(str).to_dict();s["venue_code"]=s.race_id.map(race_venue).astype(str)
    vt=venue_table.copy();vt["venue_code"]=vt.venue_code.astype(str);idx=vt.set_index("venue_code")
    def getadj(metric):
        col=metric+"_adj"
        if col not in idx:return np.zeros(len(s))
        return s.venue_code.map(idx[col]).fillna(0).to_numpy(float)
    back_win=getadj("back_win");back_top2=getadj("back_top2");bandte_win=getadj("bandte_win");bandte_top2=getadj("bandte_top2");third=getadj("third_top3");cross=getadj("otherwin_backline_second");mak=getadj("win_makuri");same=getadj("same_line_top2")
    s["v_first_back_win"]=s.first_is_back.to_numpy()*back_win
    s["v_second_back_top2"]=s.second_is_back.to_numpy()*back_top2
    s["v_first_bandte_win"]=s.first_is_bandte_of_back.to_numpy()*bandte_win
    s["v_second_bandte_top2"]=s.second_is_bandte_of_back.to_numpy()*bandte_top2
    s["v_second_third_top3"]=s.second_is_third_of_back.to_numpy()*third
    s["v_first_makuri"]=s.first_other_leader.to_numpy()*mak
    second_backline=np.maximum.reduce([s.second_is_back.to_numpy(),s.second_is_bandte_of_back.to_numpy(),s.second_is_third_of_back.to_numpy()])
    s["v_crossline_backline_second"]=s.first_other_leader.to_numpy()*second_backline*cross
    s["v_same_line_top2"]=s.same_line.to_numpy()*same
    return s


def fit_pair(scenarios: pd.DataFrame, results: pd.DataFrame):
    hit=results.loc[pd.to_numeric(results.finish_order,errors="coerce").isin([1,2])].pivot_table(index="race_id",columns="finish_order",values="car_no",aggfunc="first").rename(columns={1:"winner",2:"second"}).reset_index()
    d=scenarios.merge(hit,on="race_id",how="left");d["is_hit"]=(d.first_car.eq(d.winner)&d.second_car.eq(d.second)).astype(int)
    actual=results.loc[pd.to_numeric(results.actual_back,errors="coerce").eq(1),["race_id","car_no"]].rename(columns={"car_no":"actual_back_car"})
    actual=actual.sort_values(["race_id","actual_back_car"]).drop_duplicates("race_id")
    d=d.merge(actual,on="race_id",how="left");d=d[d.scenario_back_car.eq(d.actual_back_car)].copy()
    m=LogisticRegression(C=.25,max_iter=600,class_weight="balanced",random_state=SEED)
    m.fit(d[PAIR_FEATURES_PLUS].fillna(0),d.is_hit)
    return m


def predict_pairs(scenarios: pd.DataFrame, model):
    d=scenarios.copy();raw=model.predict_proba(d[PAIR_FEATURES_PLUS].fillna(0))[:,1];d["raw"]=raw
    den=d.groupby(["race_id","scenario_back_car"]).raw.transform("sum");d["scenario_pair_probability"]=d.raw/den
    d["weighted_probability"]=d.scenario_weight*d.scenario_pair_probability
    out=d.groupby(["race_id","first_car","second_car"],as_index=False).agg(pair_probability=("weighted_probability","sum"))
    den2=out.groupby("race_id").pair_probability.transform("sum");out.pair_probability=out.pair_probability/den2
    return out


def calibrate(valid: pd.DataFrame, targets: list[pd.DataFrame]):
    iso=IsotonicRegression(out_of_bounds="clip",y_min=1e-6,y_max=.95).fit(valid.pair_probability,valid.is_hit)
    for d in [valid]+targets:
        d["cal_raw"]=iso.predict(d.pair_probability);den=d.groupby("race_id").cal_raw.transform("sum");d["calibrated_probability"]=d.cal_raw/den
    return iso


def path_metrics(bets: pd.DataFrame,total_races:int):
    if bets.empty:
        return {"target_races":int(total_races),"purchase_races":0,"purchase_rate":0.0,"bets":0,"avg_points":0.0,"hits":0,"race_hit_rate":0.0,"stake_yen":0.0,"return_yen":0.0,"roi":0.0,"max_losing_streak":0,"max_drawdown_yen":0.0,"odds_20pct_worse_roi":0.0,"largest_hit_removed_roi":0.0}
    b=bets.sort_values(["race_date","race_id","ev"],ascending=[True,True,False]).copy();stake=float(b.stake.sum());ret=float(b["return"].sum())
    rp=b.groupby(["race_date","race_id"],sort=True).apply(lambda x:float(x["return"].sum()-x.stake.sum()),include_groups=False)
    streak=mx=0
    for loss in rp.lt(0): streak=streak+1 if loss else 0;mx=max(mx,streak)
    eq=rp.cumsum().to_numpy();peak=np.maximum.accumulate(np.r_[0.0,eq])[1:];dd=float(np.max(peak-eq)) if len(eq) else 0.0
    largest=float(b["return"].max()) if len(b) else 0.0
    return {"target_races":int(total_races),"purchase_races":int(b.race_id.nunique()),"purchase_rate":float(b.race_id.nunique()/total_races if total_races else 0),"bets":int(len(b)),"avg_points":float(len(b)/b.race_id.nunique()),"hits":int(b.is_hit.sum()),"race_hit_rate":float(b.groupby("race_id").is_hit.max().mean()),"stake_yen":stake,"return_yen":ret,"roi":ret/stake if stake else 0.0,"max_losing_streak":int(mx),"max_drawdown_yen":dd,"odds_20pct_worse_roi":ret*.8/stake if stake else 0.0,"largest_hit_removed_roi":(ret-largest)/stake if stake else 0.0}


def run_pair_variant(name, d, races25, results25, odds25, races26, results26, odds26, venue_table):
    component=d[d.race_date.le(DISC_END)].copy();pair=d[(d.race_date.gt(DISC_END))&(d.race_date.le(PAIR_END))].copy();valid=d[(d.race_date.gt(PAIR_END))&(d.race_date.le(VALID_END))].copy();test=d[d.race_date.between(TEST_START,TEST_END)].copy();ext=d[d.race_date.between(EXT_START,EXT_END)].copy()
    cats,_=base.category_effects(component);base.add_component_predictions([component,pair,valid,test,ext],component,cats)
    events=base.race_event_table(d);train_ev=events[events.race_date.le(DISC_END)]
    em=LogisticRegression(max_iter=500,class_weight="balanced",random_state=SEED).fit(train_ev[base.EVENT_FEATURES].fillna(0),train_ev.early_battle_label)
    maps={}
    for nm,f in [("pair",pair),("valid",valid),("test",test),("ext",ext)]:
        e=events[events.race_id.isin(f.race_id.unique())].copy();maps[nm]=dict(zip(e.race_id,em.predict_proba(e[base.EVENT_FEATURES].fillna(0))[:,1]))
    pm=fit_pair(scenario_rows(pair,maps["pair"],venue_table),results25)
    pv=predict_pairs(scenario_rows(valid,maps["valid"],venue_table),pm);pt=predict_pairs(scenario_rows(test,maps["test"],venue_table),pm);pe=predict_pairs(scenario_rows(ext,maps["ext"],venue_table),pm)
    vm=base.attach_market(pv,odds25,races25,results25);tm=base.attach_market(pt,odds25,races25,results25);xm=base.attach_market(pe,odds26,races26,results26)
    calibrate(vm,[tm,xm]);_,_,joint=base.reliability_tables(vm);tm=base.apply_reliability(tm,joint);xm=base.apply_reliability(xm,joint)
    _,tb=base.portfolio(tm,**FROZEN_RULE);_,xb=base.portfolio(xm,**FROZEN_RULE)
    return {"name":name,"test_market":tm,"ext_market":xm,"test_bets":tb,"ext_bets":xb,"test":path_metrics(tb,test.race_id.nunique()),"external":path_metrics(xb,ext.race_id.nunique())}


def raw_metric_table(d:pd.DataFrame,races:pd.DataFrame):
    rows=[]
    for metric in METRICS:
        f,_=metric_frame(metric,d,races)
        f=f[f.race_date.between(20250101,EXT_END)].copy();f["split"]=f.race_date.astype(int).map(split_name)
        for (split,vc,vn),g in f.groupby(["split","venue_code","venue_name"]):
            if split=="other":continue
            rows.append({"split":split,"venue_code":str(vc).zfill(2),"venue_name":vn,"metric":metric,"metric_ja":METRIC_JA[metric],"n":len(g),"rate":float(g.y.mean())})
    return pd.DataFrame(rows)


def stratified_table(d:pd.DataFrame,races:pd.DataFrame):
    defs=[("formation",["二分戦","三分戦","細切れ"]),("starter_group",["7車","9車"]),("class_group",["A級","S級"])]
    rows=[]
    for dim,vals in defs:
        for val in vals:
            sub=d[d[dim].eq(val)];rr=races[races[dim].eq(val)]
            for metric in ["back_win","back_second","bandte_win","bandte_top2","third_top3","otherwin_backline_second","front_selfpower_second","win_escape","win_makuri","win_sashi","same_line_top2"]:
                f,_=metric_frame(metric,sub,rr)
                for (vc,vn),g in f.groupby(["venue_code","venue_name"]):
                    if len(g):rows.append({"venue_code":str(vc).zfill(2),"venue_name":vn,"dimension":dim,"segment":val,"metric":metric,"metric_ja":METRIC_JA[metric],"n":len(g),"rate":float(g.y.mean())})
    return pd.DataFrame(rows)


def classify_effects(effects:pd.DataFrame,resid:pd.DataFrame,roi_gate:bool):
    merged=effects.copy()
    for split,prefix in [("locked_test_2025_11_12","test"),("external_2026_01_02","external")]:
        z=resid[resid.split.eq(split)][["venue_code","metric","n","residual_pp","ci95_low_pp","ci95_high_pp"]].copy().rename(columns={"n":prefix+"_n","residual_pp":prefix+"_residual_pp","ci95_low_pp":prefix+"_ci_low_pp","ci95_high_pp":prefix+"_ci_high_pp"})
        merged=merged.merge(z,on=["venue_code","metric"],how="left")
    def cls(r):
        rd=r.discovery_residual_pp;rt=r.get("test_residual_pp",np.nan);re=r.get("external_residual_pp",np.nan)
        enough_hold=(r.get("test_n",0)>=MIN_TEST_N and r.get("external_n",0)>=MIN_EXT_N)
        same_hold=np.isfinite(rt) and np.isfinite(re) and rd*rt>0 and rd*re>0
        if r.candidate_reproduced and enough_hold and same_hold:return "本当に再現した癖"
        if r.candidate_reproduced:return "弱い傾向"
        if r.discovery_n>=MIN_DISC_N and np.isfinite(rd) and abs(rd)>=1.0:return "再現しなかった一般論"
        return "補正根拠なし"
    merged["classification"]=merged.apply(cls,axis=1)
    merged["use_in_prediction"]=(merged.classification.eq("本当に再現した癖") & merged.metric.isin({"back_win","back_top2","bandte_win","bandte_top2","third_top3","otherwin_backline_second","win_makuri","same_line_top2"}) & roi_gate)
    return merged


def make_cards(classified:pd.DataFrame,raw:pd.DataFrame,roi_gate:bool):
    venues=classified[["venue_code","venue_name"]].drop_duplicates().sort_values("venue_code");rows=[];external_raw=raw[raw.split.eq("external_2026_01_02")]
    for _,v in venues.iterrows():
        g=classified[classified.venue_code.eq(v.venue_code)]
        def names(label):return "、".join(g.loc[g.classification.eq(label),"metric_ja"].tolist()) or "なし"
        use="、".join(g.loc[g.use_in_prediction,"metric_ja"].tolist()) or "なし";no="、".join(g.loc[~g.use_in_prediction,"metric_ja"].tolist()) or "なし";er=external_raw[external_raw.venue_code.eq(v.venue_code)]
        def rate(m):
            z=er[er.metric.eq(m)];return float(z.rate.iloc[0]) if len(z) else None
        rows.append({"venue_code":v.venue_code,"venue_name":v.venue_name,"true_reproduced":names("本当に再現した癖"),"weak_tendency":names("弱い傾向"),"failed_generalization":names("再現しなかった一般論"),"prediction_adjustments":use if roi_gate else "なし（ROIゲート不通過）","no_adjustment_items":no,"external_back_win":rate("back_win"),"external_back_second":rate("back_second"),"external_back_third":rate("back_third"),"external_bandte_win":rate("bandte_win"),"external_bandte_top2":rate("bandte_top2"),"external_bandte_top3":rate("bandte_top3"),"external_third_top3":rate("third_top3"),"external_otherwin_backline_second":rate("otherwin_backline_second"),"external_front_selfpower_second":rate("front_selfpower_second"),"external_escape":rate("win_escape"),"external_makuri":rate("win_makuri"),"external_sashi":rate("win_sashi"),"external_same_line_top2":rate("same_line_top2")})
    return pd.DataFrame(rows)


def fmt(x):
    try:
        if x is None or pd.isna(x):return "NA"
        return f"{float(x)*100:.1f}%"
    except Exception:return "NA"


def markdown_report(cards,roi_df,venue_count,roi_gate):
    lines=["# 全国競輪場・実測攻略レポート","",f"対象場数: **{venue_count}場**","", "## 検証設計", "", "- 2025年1〜6月: 場差候補の発見・能力/展開ベースライン", "- 2025年7〜8月: 2車単順序モデル学習", "- 2025年9〜10月: 場差の別期間再現確認、確率校正", "- 2025年11〜12月: 場補正あり/なしROIゲート", "- 2026年1〜2月: 完全未使用の最終外部比較（調整禁止）", "", "場差は選手能力、級班、B/H、ライン位置・人数、二/三分戦、車立て等を入れたvenue無しモデルの残差で評価。発見期間と確認期間で符号が再現し、母数条件を満たす候補だけを場補正候補にした。", "", "## 2車単ROI比較", ""]
    for _,r in roi_df.iterrows():lines.append(f"- {r['period']} / {r['variant']}: ROI **{r['roi']*100:.2f}%**, 購入率 {r['purchase_rate']*100:.2f}%, 的中率 {r['race_hit_rate']*100:.2f}%, 最大連敗 {int(r['max_losing_streak'])}, 最大DD ¥{r['max_drawdown_yen']:.0f}")
    lines += ["",f"**最終場補正採用: {'採用' if roi_gate else '不採用'}**","", "## 各場カード",""]
    for _,c in cards.iterrows():
        lines += [f"### {c.venue_name}",f"- ① 本当に再現した癖: {c.true_reproduced}",f"- ② 弱い傾向: {c.weak_tendency}",f"- ③ 再現しなかった一般論: {c.failed_generalization}",f"- ④ 予想時に使う補正: {c.prediction_adjustments}",f"- ⑤ 補正しない項目: {c.no_adjustment_items}",f"- 未使用2026実測: B取得者 1着 {fmt(c.external_back_win)} / 2着 {fmt(c.external_back_second)} / 3着 {fmt(c.external_back_third)}、番手 1着 {fmt(c.external_bandte_win)} / 2連対 {fmt(c.external_bandte_top2)} / 3連対 {fmt(c.external_bandte_top3)}、3番手3着内 {fmt(c.external_third_top3)}",f"- 展開系: 別線1着時Bライン2着残り {fmt(c.external_otherwin_backline_second)}、前で踏んだ自力2着 {fmt(c.external_front_selfpower_second)}、逃 {fmt(c.external_escape)} / 捲 {fmt(c.external_makuri)} / 差 {fmt(c.external_sashi)}、1-2着同ライン {fmt(c.external_same_line_top2)}",""]
    lines += ["## 定義", "", "- B取得者: 公式結果 `actual_back=1` の選手。", "- B取得ライン番手/3番手: 実際のB取得者と同じラインの事前並び2番手/3番手。", "- 前で踏んだ自力: ライン先頭かつ自力型で、公式SまたはBを取得した選手。", "- ライン丸ごと残る率: 主指標は1-2着同ライン率。3車ライン上位3着独占率も別CSVに保存。", "- 二分戦/三分戦/細切れ、7/9車、A/S級の場別クロス集計は `venue_stratified_metrics.csv`。"]
    return "\n".join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--train-dir",required=True);ap.add_argument("--external-dir",required=True);ap.add_argument("--output-dir",default="venue_analysis");a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    r25,u25,y25,o25=load(Path(a.train_dir),"2025");r26,u26,y26,o26=load(Path(a.external_dir),"2026_01_02")
    r26=r26[r26.race_date.between(EXT_START,EXT_END)].copy();ids=set(r26.race_id);u26=u26[u26.race_id.isin(ids)].copy();y26=y26[y26.race_id.isin(ids)].copy();o26=o26[o26.race_id.isin(ids)].copy()
    d=prepare_rider_frame(r25,u25,y25,r26,u26,y26);rf=make_race_frame(d)
    raw=raw_metric_table(d,rf);resid,model_meta=venue_metric_residuals(d,rf);effects,venue_table=build_adjustment_table(resid)
    codes=d[["venue_code","venue_name"]].drop_duplicates().copy();codes["venue_code"]=codes.venue_code.astype(str).str.zfill(2);venue_table["venue_code"]=venue_table.venue_code.astype(str).str.zfill(2)
    zero=zero_venue_table(codes)
    novenue=run_pair_variant("場補正なし",d,r25,y25,o25,r26,y26,o26,zero)
    venue=run_pair_variant("再現確認済み場補正",d,r25,y25,o25,r26,y26,o26,venue_table)
    roi_rows=[]
    for period,key in [("2025-11..12 ROIゲート","test"),("2026-01..02 未使用外部","external")]:
        for v in [novenue,venue]:roi_rows.append({"period":period,"variant":v["name"],**v[key]})
    roi=pd.DataFrame(roi_rows)
    test_base=novenue["test"]["roi"];test_venue=venue["test"]["roi"];ext_base=novenue["external"]["roi"];ext_venue=venue["external"]["roi"]
    roi_gate=bool(test_venue>test_base and ext_venue>ext_base)
    classified=classify_effects(effects,resid,roi_gate);cards=make_cards(classified,raw,roi_gate);strat=stratified_table(d,rf)

    raw.to_csv(out/"venue_raw_metrics.csv",index=False,encoding="utf-8-sig");resid.to_csv(out/"venue_adjusted_residuals.csv",index=False,encoding="utf-8-sig");effects.to_csv(out/"venue_candidate_effects.csv",index=False,encoding="utf-8-sig");classified.to_csv(out/"venue_effect_classification.csv",index=False,encoding="utf-8-sig");venue_table.to_csv(out/"venue_correction_table_prelocked.csv",index=False,encoding="utf-8-sig");strat.to_csv(out/"venue_stratified_metrics.csv",index=False,encoding="utf-8-sig");cards.to_csv(out/"venue_cards.csv",index=False,encoding="utf-8-sig");roi.to_csv(out/"venue_roi_comparison.csv",index=False,encoding="utf-8-sig");model_meta.to_csv(out/"venue_residual_model_meta.csv",index=False,encoding="utf-8-sig")
    novenue["test_bets"].to_csv(out/"bets_2025_test_no_venue.csv",index=False,encoding="utf-8-sig");venue["test_bets"].to_csv(out/"bets_2025_test_with_venue.csv",index=False,encoding="utf-8-sig");novenue["ext_bets"].to_csv(out/"bets_2026_external_no_venue.csv",index=False,encoding="utf-8-sig");venue["ext_bets"].to_csv(out/"bets_2026_external_with_venue.csv",index=False,encoding="utf-8-sig")

    venue_count=int(cards.venue_code.nunique());report={"execution":"Python actual run","dataset":{"2025_races":int(r25.race_id.nunique()),"2026_01_02_races":int(r26.race_id.nunique()),"venues_observed":venue_count},"time_split":{"venue_discovery":"2025-01-01..2025-06-30","pair_train":"2025-07-01..2025-08-31","venue_validation_and_calibration":"2025-09-01..2025-10-31","roi_gate":"2025-11-01..2025-12-31","external_untouched":"2026-01-01..2026-02-28"},"confounding_control":"venue excluded HGB baseline with ability/rank/B/H/line composition/starters/class/day/distance/wind and relative features; venue effect is residual actual-minus-expected","candidate_rules":{"min_discovery_n":MIN_DISC_N,"min_validation_n":MIN_VALID_N,"same_sign_required":True,"min_abs_discovery_pp":0.75,"min_abs_validation_pp":0.50,"shrink_k":SHRINK_K,"max_logit_adjustment":MAX_ADJ},"frozen_purchase_rule":FROZEN_RULE,"roi_comparison":roi_rows,"roi_gate_passed":roi_gate,"production_venue_correction":"ADOPT" if roi_gate else "REJECT","important":"2026-01..02 labels were never used to estimate venue adjustments, choose candidate venue-metrics, fit pair model, calibrate probabilities, or set purchase thresholds. They are only the final external comparison.","files":["venue_cards.csv","venue_raw_metrics.csv","venue_adjusted_residuals.csv","venue_effect_classification.csv","venue_stratified_metrics.csv","venue_roi_comparison.csv"]}
    (out/"venue_strategy_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");(out/"venue_strategy_report.md").write_text(markdown_report(cards,roi,venue_count,roi_gate),encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
