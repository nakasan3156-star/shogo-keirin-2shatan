#!/usr/bin/env python3
"""前走の「負けて強し／展開不利」パターンが次走で再現するか実測する。

ラベルは必ず前走結果・前走コメントだけから作る。次走結果を見て定義を変えない。
2025年前半=発見、2025年後半=再現確認、2026年1-2月=外部確認。
能力差は次走時点のvenue/前走ラベルを含まない能力モデルの期待値との差で除く。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

SEED = 3156
DISC_END = 20250630
VALID_END = 20251231
EXT_START = 20260101
EXT_END = 20260228
MIN_DISC_N = 80
MIN_VALID_N = 80
MIN_EXT_N = 25

LABELS = {
    "back_4plus": "B取得して4着以下",
    "back_5plus": "B取得して5着以下",
    "start_back_4plus": "S・B両方取得して4着以下",
    "leader_self_back_4plus": "ライン先頭の自力がB取得して4着以下",
    "back_4plus_bandte_top3": "B取得4着以下だが番手が3着内",
    "back_4plus_line_top3": "B取得4着以下だが同ライン選手が3着内",
    "back_4plus_otherline_win": "B取得4着以下で別線が1着",
    "lap_top3_4plus": "上がり3位以内なのに4着以下",
    "back_lap_top3_4plus": "B取得＋上がり3位以内で4着以下",
    "battle_4plus": "先行争い・叩き合いで4着以下",
    "long_drive_4plus": "先行・逃げ・突張・カマシで4着以下",
    "bandte_fight_4plus": "競り・番手飛ばされ等で4着以下",
    "target_failed_4plus": "前・目標不発で4着以下",
    "rear_4plus": "後方・後手で4着以下",
    "blocked_4plus": "牽制・進路・詰まり等で4着以下",
    "position_cost_4plus": "位置取り・追上げ・脚使いで4着以下",
    "restart_4plus": "再仕掛け・立て直しで4着以下",
    "legs_left_4plus": "脚余し・仕掛け遅れ・届かずで4着以下",
    "accident_prev": "前走事故あり",
}

ABILITY_FEATURES = [
    "score", "s_count", "h_count", "b_count", "escape", "makuri", "sashi", "mark",
    "finish_1", "finish_2", "finish_3", "finish_out", "win_rate", "top2_rate", "top3_rate",
    "actual_starters", "line_count", "line_position", "line_size", "rank_level", "style_code",
    "recent5_avg_order", "recent5_back_rate", "recent5_standing_rate", "recent5_avg_lap",
    "score_rel", "b_rel", "h_rel", "escape_rel", "makuri_rel", "sashi_rel",
    "line_score_rel", "line_b_rel", "is_leader", "is_bandte", "is_third", "is_self_power",
]


def load(root: Path, label: str):
    races = pd.read_csv(root / f"races_{label}.csv", dtype={"race_id": str, "venue_code": str})
    riders = pd.read_csv(root / f"rider_features_{label}.csv", dtype={"race_id": str, "venue_code": str, "player_id": str})
    results = pd.read_csv(root / f"official_results_{label}.csv", dtype={"race_id": str, "player_id": str})
    for d in (races, riders, results):
        for c in ["race_date","car_no","finish_order","actual_back","actual_start","final_lap_time","line_no","line_position","line_size","actual_starters","line_count"]:
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
    return races, riders, results


def enrich(races: pd.DataFrame, riders: pd.DataFrame, results: pd.DataFrame):
    keep = ["race_id","car_no","player_id","finish_order","winning_move","actual_start","actual_back","final_lap_time","has_accident","accident_name","result_comment"]
    x = riders.merge(results[keep], on=["race_id","car_no","player_id"], how="inner")
    x = x.merge(races[["race_id","distance_m","wind_speed"]], on="race_id", how="left", suffixes=("","_race"))
    for c in ["race_date","score","s_count","h_count","b_count","escape","makuri","sashi","mark","finish_1","finish_2","finish_3","finish_out","win_rate","top2_rate","top3_rate","recent5_avg_order","recent5_back_rate","recent5_standing_rate","recent5_avg_lap"]:
        if c in x: x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[x.finish_order.gt(0)].copy()
    x["rank_level"] = x["rank"].map({"A3":1,"A2":2,"A1":3,"S2":4,"S1":5,"SS":6}).fillna(0)
    x["style_code"] = x["style"].map({"逃":3,"両":2,"追":1}).fillna(0)
    g = x.groupby("race_id", sort=False)
    for s,t in [("score","score_rel"),("b_count","b_rel"),("h_count","h_rel"),("escape","escape_rel"),("makuri","makuri_rel"),("sashi","sashi_rel")]:
        x[t] = x[s] - g[s].transform("mean")
    line = x.groupby(["race_id","line_no"], sort=False)
    x["line_score"] = line.score.transform("sum"); x["line_b"] = line.b_count.transform("sum")
    x["line_score_rel"] = x.line_score - g.line_score.transform("mean")
    x["line_b_rel"] = x.line_b - g.line_b.transform("mean")
    x["is_leader"] = x.line_position.eq(1).astype(int); x["is_bandte"] = x.line_position.eq(2).astype(int); x["is_third"] = x.line_position.eq(3).astype(int)
    x["is_self_power"] = ((x.line_position.eq(1)) | (x.b_count.gt(0)) | ((x.escape+x.makuri).gt(0))).astype(int)
    # 上がり順位。0秒・欠測は順位対象外。
    lap = pd.to_numeric(x.final_lap_time, errors="coerce").where(lambda z:z.gt(0))
    x["lap_rank"] = lap.groupby(x.race_id).rank(method="min", ascending=True)
    # 勝者ライン、B取得者ライン内の残り。
    winner_line = x.loc[x.finish_order.eq(1), ["race_id","line_no"]].drop_duplicates("race_id").set_index("race_id").line_no
    x["winner_line"] = x.race_id.map(winner_line)
    top3_line = x.loc[x.finish_order.le(3), ["race_id","line_no","line_position"]].copy()
    bandte_top3 = set(map(tuple, top3_line.loc[top3_line.line_position.eq(2), ["race_id","line_no"]].to_numpy()))
    line_top3 = set(map(tuple, top3_line[["race_id","line_no"]].drop_duplicates().to_numpy()))
    x["bandte_same_line_top3"] = [((r,l) in bandte_top3) for r,l in zip(x.race_id,x.line_no)]
    x["same_line_top3"] = [((r,l) in line_top3) for r,l in zip(x.race_id,x.line_no)]
    return x


def add_loss_labels(x: pd.DataFrame):
    lost = x.finish_order.ge(4)
    back = x.actual_back.fillna(0).eq(1)
    start = x.actual_start.fillna(0).eq(1)
    comment = x.result_comment.fillna("").astype(str)
    x["back_4plus"] = back & lost
    x["back_5plus"] = back & x.finish_order.ge(5)
    x["start_back_4plus"] = start & back & lost
    x["leader_self_back_4plus"] = x.is_leader.eq(1) & x.is_self_power.eq(1) & back & lost
    x["back_4plus_bandte_top3"] = back & lost & x.bandte_same_line_top3
    x["back_4plus_line_top3"] = back & lost & x.same_line_top3
    x["back_4plus_otherline_win"] = back & lost & x.winner_line.ne(x.line_no)
    x["lap_top3_4plus"] = x.lap_rank.le(3) & lost
    x["back_lap_top3_4plus"] = back & x.lap_rank.le(3) & lost
    pats = {
        "battle_4plus": r"先行争|叩き合|踏み合|叩き叩かれ|もがき合",
        "long_drive_4plus": r"先行|逃げ|突張|突っ張|カマシ|ペース駆け|正攻法逃げ",
        "bandte_fight_4plus": r"番手飛|競り|競勝|番手奪|競負|競負け",
        "target_failed_4plus": r"前不発|目標が不発|目標共倒れ|不発ライン|前が不発|目標不発",
        "rear_4plus": r"後方置かれ|最後方|後方|後手ライン|後手",
        "blocked_4plus": r"牽制|進路|詰まり|阻ま|張られ|捌かれ|包ま|コース無|コースなく",
        "position_cost_4plus": r"位置取|追上|追い上|斬り込|脚使|踏まされ",
        "restart_4plus": r"再仕掛|立て直|仕掛け直",
        "legs_left_4plus": r"脚余|仕掛け遅|余し|届かず|届かない|届かぬ",
    }
    for key,pat in pats.items(): x[key] = lost & comment.str.contains(pat, regex=True)
    x["accident_prev"] = pd.to_numeric(x.has_accident,errors="coerce").fillna(0).eq(1)
    return x


def next_start_rows(x: pd.DataFrame):
    x = x.sort_values(["player_id","race_date","race_id"]).copy()
    # 現在行が「次走」。前走ラベルをplayer_id内で1つシフト。
    for key in LABELS:
        x["prev_"+key] = x.groupby("player_id")[key].shift(1).fillna(False).astype(bool)
    x["prev_race_id"] = x.groupby("player_id").race_id.shift(1)
    x["prev_race_date"] = x.groupby("player_id").race_date.shift(1)
    x["prev_finish"] = x.groupby("player_id").finish_order.shift(1)
    x["prev_comment"] = x.groupby("player_id").result_comment.shift(1)
    dates = pd.to_datetime(x.race_date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    prev_dates = pd.to_datetime(pd.to_numeric(x.prev_race_date,errors="coerce").astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    x["days_since_prev"] = (dates-prev_dates).dt.days
    x = x[x.prev_race_id.notna() & x.days_since_prev.gt(0)].copy()
    x["y_win"] = x.finish_order.eq(1).astype(int); x["y_top2"] = x.finish_order.le(2).astype(int); x["y_top3"] = x.finish_order.le(3).astype(int)
    return x


def fit_model(df, label):
    m = HistGradientBoostingClassifier(learning_rate=.05,max_iter=120,max_leaf_nodes=23,min_samples_leaf=60,l2_regularization=3,random_state=SEED)
    m.fit(df[ABILITY_FEATURES].fillna(-99),df[label].astype(int)); return m


def add_expected(df: pd.DataFrame):
    disc = df[df.race_date.le(DISC_END)].copy().reset_index()
    if disc.empty: raise RuntimeError("no discovery rows")
    groups = disc.race_id.astype(str)
    for y,name in [("y_win","exp_win"),("y_top2","exp_top2"),("y_top3","exp_top3")]:
        oof = np.zeros(len(disc)); gkf=GroupKFold(n_splits=3)
        for tr,va in gkf.split(disc,disc[y],groups):
            model=fit_model(disc.iloc[tr],y);oof[va]=model.predict_proba(disc.iloc[va][ABILITY_FEATURES].fillna(-99))[:,1]
        full=fit_model(disc,y);df[name]=full.predict_proba(df[ABILITY_FEATURES].fillna(-99))[:,1]
        df.loc[disc["index"],name]=oof
    return df


def split_name(d):
    if d<=DISC_END:return "discovery_2025H1"
    if d<=VALID_END:return "validation_2025H2"
    if EXT_START<=d<=EXT_END:return "external_2026_01_02"
    return "other"


def stats(df: pd.DataFrame):
    rows=[]
    df=df.copy();df["split"]=df.race_date.astype(int).map(split_name)
    for key,ja in LABELS.items():
        mask=df["prev_"+key]
        for split,g in df[mask].groupby("split"):
            if split=="other":continue
            n=len(g)
            rows.append({"label":key,"label_ja":ja,"split":split,"n":n,
                "win_rate":float(g.y_win.mean()),"top2_rate":float(g.y_top2.mean()),"top3_rate":float(g.y_top3.mean()),
                "expected_win":float(g.exp_win.mean()),"expected_top2":float(g.exp_top2.mean()),"expected_top3":float(g.exp_top3.mean()),
                "win_residual_pp":float((g.y_win-g.exp_win).mean()*100),
                "top2_residual_pp":float((g.y_top2-g.exp_top2).mean()*100),
                "top3_residual_pp":float((g.y_top3-g.exp_top3).mean()*100),
                "median_days_to_next":float(g.days_since_prev.median()),
            })
    return pd.DataFrame(rows)


def classify(s: pd.DataFrame):
    rows=[]
    for key,ja in LABELS.items():
        z=s[s.label.eq(key)].set_index("split")
        def get(split,col,default=np.nan):
            return z.at[split,col] if split in z.index else default
        nd=get("discovery_2025H1","n",0);nv=get("validation_2025H2","n",0);ne=get("external_2026_01_02","n",0)
        rd=get("discovery_2025H1","top2_residual_pp");rv=get("validation_2025H2","top2_residual_pp");re=get("external_2026_01_02","top2_residual_pp")
        candidate=bool(nd>=MIN_DISC_N and nv>=MIN_VALID_N and np.isfinite(rd) and np.isfinite(rv) and rd>1.0 and rv>0.5)
        true=bool(candidate and ne>=MIN_EXT_N and np.isfinite(re) and re>0)
        danger=bool(nd>=MIN_DISC_N and nv>=MIN_VALID_N and ne>=MIN_EXT_N and np.isfinite(rd) and np.isfinite(rv) and np.isfinite(re) and rd<0 and rv<0 and re<0)
        if true:cl="本物の負けて強し／展開不利"
        elif candidate:cl="2025では再現・外部未確認"
        elif danger:cl="次走危険パターン"
        elif nd>=MIN_DISC_N and np.isfinite(rd) and rd>1:cl="前半だけ・再現せず"
        else:cl="有効性なし／母数不足"
        rows.append({"label":key,"label_ja":ja,"classification":cl,"discovery_n":int(nd),"validation_n":int(nv),"external_n":int(ne),
            "discovery_top2_residual_pp":rd,"validation_top2_residual_pp":rv,"external_top2_residual_pp":re,
            "external_win_rate":get("external_2026_01_02","win_rate"),"external_top2_rate":get("external_2026_01_02","top2_rate"),"external_top3_rate":get("external_2026_01_02","top3_rate"),
            "external_win_residual_pp":get("external_2026_01_02","win_residual_pp"),"external_top3_residual_pp":get("external_2026_01_02","top3_residual_pp"),
            "median_days_to_next_external":get("external_2026_01_02","median_days_to_next"),
        })
    return pd.DataFrame(rows).sort_values(["classification","external_top2_residual_pp"],ascending=[True,False])


def examples(df: pd.DataFrame, classified: pd.DataFrame):
    good=set(classified.loc[classified.classification.eq("本物の負けて強し／展開不利"),"label"])
    rows=[]
    for key in good:
        g=df[df["prev_"+key] & df.race_date.between(EXT_START,EXT_END) & df.finish_order.le(3)].sort_values(["finish_order","race_date"]).head(20)
        for _,r in g.iterrows():
            rows.append({"label":key,"label_ja":LABELS[key],"player_id":r.player_id,"name":r.get("name",""),"prev_race_date":r.prev_race_date,"prev_finish":r.prev_finish,"prev_comment":r.prev_comment,"next_race_date":r.race_date,"next_venue":r.venue_name,"next_finish":r.finish_order,"days_since_prev":r.days_since_prev})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--train-dir",required=True);ap.add_argument("--external-dir",required=True);ap.add_argument("--output-dir",default="next_start_analysis");a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    r25,u25,y25=load(Path(a.train_dir),"2025");r26,u26,y26=load(Path(a.external_dir),"2026_01_02")
    r26=r26[r26.race_date.between(EXT_START,EXT_END)].copy();ids=set(r26.race_id);u26=u26[u26.race_id.isin(ids)];y26=y26[y26.race_id.isin(ids)]
    races=pd.concat([r25,r26],ignore_index=True,sort=False);riders=pd.concat([u25,u26],ignore_index=True,sort=False);results=pd.concat([y25,y26],ignore_index=True,sort=False)
    x=add_loss_labels(enrich(races,riders,results));nxt=next_start_rows(x);nxt=add_expected(nxt)
    s=stats(nxt);c=classify(s);e=examples(nxt,c)
    s.to_csv(out/"next_start_pattern_by_split.csv",index=False,encoding="utf-8-sig");c.to_csv(out/"next_start_pattern_classification.csv",index=False,encoding="utf-8-sig");e.to_csv(out/"external_examples.csv",index=False,encoding="utf-8-sig")
    # 全ラベルの前走コメント例も監査用に保存。
    audit=[]
    for key,ja in LABELS.items():
        q=x[x[key]][["race_id","race_date","player_id","name","finish_order","actual_start","actual_back","result_comment"]].head(50).copy();q.insert(0,"label_ja",ja);q.insert(0,"label",key);audit.append(q)
    pd.concat(audit,ignore_index=True).to_csv(out/"label_comment_audit.csv",index=False,encoding="utf-8-sig")
    report={"execution":"Python actual run","dataset":{"2025_races":int(r25.race_id.nunique()),"2026_external_races":int(r26.race_id.nunique()),"next_start_rows":int(len(nxt)),"players":int(nxt.player_id.nunique())},"definition":"labels use previous race only; current/next result never enters label definition","ability_control":"next-start expected win/top2/top3 from venue-free ability/context HGB excluding previous-result labels; effects are actual-minus-expected residual","splits":{"discovery":"next starts in 2025-01..06","validation":"next starts in 2025-07..12","external":"next starts in 2026-01..02"},"criteria":{"discovery_n":MIN_DISC_N,"validation_n":MIN_VALID_N,"external_n":MIN_EXT_N,"primary":"top2 residual","candidate":"discovery > +1.0pp and validation > +0.5pp","true":"candidate plus external residual >0"},"classification_counts":c.classification.value_counts().to_dict(),"true_patterns":c[c.classification.eq("本物の負けて強し／展開不利")].to_dict("records"),"danger_patterns":c[c.classification.eq("次走危険パターン")].to_dict("records")}
    (out/"next_start_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
