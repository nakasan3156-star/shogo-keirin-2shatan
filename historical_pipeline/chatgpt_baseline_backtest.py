#!/usr/bin/env python3
"""ChatGPT式: 数字→展開→恩恵→場補正→着内率→2車単 の基準モデル。

オッズは全確率を確定した後の購入判定だけで参照する。選手名・player_idは特徴量に使わない。
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SEED = 20250810
COMPONENT_END = 20250630
PAIR_END = 20250831
VALID_END = 20251031
TEST_START = 20251101

NUMERIC_BASE = [
    "score", "escape", "makuri", "sashi", "mark", "b_count", "h_count", "s_count",
    "finish_1", "finish_2", "finish_3", "finish_out", "win_rate", "top2_rate", "top3_rate",
    "car_no", "scheduled_starters", "actual_starters", "line_count", "line_position", "line_size",
    "rank_level", "style_code", "day_no", "distance_m", "wind_speed",
]
RELATIVE = [
    "score_rel", "b_rel", "h_rel", "escape_rel", "makuri_rel", "sashi_rel", "mark_rel",
    "score_rank", "b_rank", "h_rank", "escape_rank", "makuri_rank",
    "b_top_gap", "h_top_gap", "score_top_gap", "line_score_rel", "line_b_rel", "line_h_rel",
    "is_leader", "is_bandte", "is_third", "is_single", "is_self_power",
    "two_line", "three_line", "fragmented", "leader_b_gap", "leader_h_gap", "escape_leader_count",
]
PRIOR = [
    "has_previous_day", "prior_A", "prior_B", "prior_C", "prior_D", "prior_E", "prior_F",
    "prior_G", "prior_H", "prior_I", "prior_finish", "prior_lap_rel", "prior_back", "prior_start",
]
FEATURES = NUMERIC_BASE + RELATIVE + PRIOR

PAIR_FEATURES = [
    "log_first_win", "log_second_top2", "first_is_back", "second_is_back",
    "first_is_bandte_of_back", "second_is_bandte_of_back", "first_is_third_of_back",
    "second_is_third_of_back", "first_other_leader", "second_other_leader", "same_line",
    "front_bandte", "bandte_front", "front_third", "third_front", "line_order",
    "battle_first_makuri", "battle_second_front", "battle_second_bandte", "score_diff",
    "first_line_position", "second_line_position", "venue_first_adj", "venue_second_adj",
]


def safe_logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -25, 25)))


def normalize_by_race(df, raw, out, total=1.0):
    den = df.groupby("race_id")[raw].transform("sum")
    cnt = df.groupby("race_id")[raw].transform("count")
    df[out] = np.where(den > 0, df[raw] * total / den, total / cnt)


def add_prerace_features(riders, races, results):
    r = riders.merge(
        races[["race_id", "start_date", "day_no", "distance_m", "wind_speed", "two_car_exacta_combination", "two_car_exacta_payout_yen"]],
        on="race_id", how="left", suffixes=("", "_race")
    )
    for c in NUMERIC_BASE + ["race_date", "start_date"]:
        if c in r: r[c] = pd.to_numeric(r[c], errors="coerce")
    r["rank_level"] = r["rank"].map({"A3": 1, "A2": 2, "A1": 3, "S2": 4, "S1": 5}).fillna(0)
    r["style_code"] = r["style"].map({"逃": 3, "両": 2, "追": 1}).fillna(0)
    g = r.groupby("race_id", sort=False)
    for s, t in [("score","score_rel"),("b_count","b_rel"),("h_count","h_rel"),("escape","escape_rel"),("makuri","makuri_rel"),("sashi","sashi_rel"),("mark","mark_rel")]:
        r[t] = r[s] - g[s].transform("mean")
    for s, t in [("score","score_rank"),("b_count","b_rank"),("h_count","h_rank"),("escape","escape_rank"),("makuri","makuri_rank")]:
        r[t] = g[s].rank(method="min", ascending=False)
    def top_gap(x):
        z = np.sort(pd.to_numeric(x, errors="coerce").fillna(0).to_numpy())[::-1]
        return float(z[0] - z[1]) if len(z) > 1 else float(z[0] if len(z) else 0)
    for s,t in [("b_count","b_top_gap"),("h_count","h_top_gap"),("score","score_top_gap")]:
        m = g[s].apply(top_gap).to_dict(); r[t] = r["race_id"].map(m)
    line = r.groupby(["race_id","line_no"], sort=False)
    r["line_score"] = line["score"].transform("sum")
    r["line_b"] = line["b_count"].transform("sum")
    r["line_h"] = line["h_count"].transform("sum")
    for s,t in [("line_score","line_score_rel"),("line_b","line_b_rel"),("line_h","line_h_rel")]:
        r[t] = r[s] - g[s].transform("mean")
    r["is_leader"] = (r["line_position"] == 1).astype(int)
    r["is_bandte"] = (r["line_position"] == 2).astype(int)
    r["is_third"] = (r["line_position"] == 3).astype(int)
    r["is_single"] = (r["line_size"] == 1).astype(int)
    r["is_self_power"] = ((r["line_position"] == 1) | (r["b_count"] > 0) | ((r["escape"] + r["makuri"]) > 0)).astype(int)
    r["two_line"] = (r["line_count"] == 2).astype(int)
    r["three_line"] = (r["line_count"] == 3).astype(int)
    r["fragmented"] = (r["line_count"] >= 4).astype(int)
    lead = r[r["is_leader"].eq(1)].copy()
    def lead_gap(frame, col):
        z=np.sort(frame[col].fillna(0).to_numpy())[::-1]
        return float(z[0]-z[1]) if len(z)>1 else float(z[0] if len(z) else 0)
    r["leader_b_gap"] = r["race_id"].map(lead.groupby("race_id").apply(lambda x: lead_gap(x,"b_count"), include_groups=False))
    r["leader_h_gap"] = r["race_id"].map(lead.groupby("race_id").apply(lambda x: lead_gap(x,"h_count"), include_groups=False))
    r["escape_leader_count"] = r["race_id"].map(lead.assign(x=(lead.escape>0).astype(int)).groupby("race_id").x.sum()).fillna(0)

    # 前日情報は同一開催の直前日だけ。現在レースの結果は特徴量に入らない。
    fact = results[["race_id","car_no","finish_order","actual_start","actual_back","final_lap_time","result_comment","winning_move"]].copy()
    fact = fact.rename(columns={c:f"cur_{c}" for c in fact.columns if c not in ["race_id","car_no"]})
    hist = r[["race_id","race_date","start_date","day_no","venue_code","player_id","car_no"]].merge(fact,on=["race_id","car_no"],how="left")
    hist = hist.sort_values(["player_id","race_date","race_id"])
    shift_cols=["start_date","venue_code","race_date","cur_finish_order","cur_actual_start","cur_actual_back","cur_final_lap_time","cur_result_comment","cur_winning_move"]
    for c in shift_cols: hist[f"prev_{c}"] = hist.groupby("player_id")[c].shift(1)
    same = (hist.day_no.gt(1) & hist.start_date.eq(hist.prev_start_date) & hist.venue_code.eq(hist.prev_venue_code) & hist.race_date.gt(hist.prev_race_date))
    hist["has_previous_day"] = same.astype(int)
    comment = hist.prev_cur_result_comment.fillna("").where(same, "")
    finish = pd.to_numeric(hist.prev_cur_finish_order,errors="coerce").where(same)
    back = pd.to_numeric(hist.prev_cur_actual_back,errors="coerce").fillna(0).where(same,0)
    start = pd.to_numeric(hist.prev_cur_actual_start,errors="coerce").fillna(0).where(same,0)
    lap = pd.to_numeric(hist.prev_cur_final_lap_time,errors="coerce").where(same)
    defs={
        "A": r"先行争|叩き合|踏み合|叩き叩かれ",
        "B": r"先行|逃げ|突張|カマシ逃げ|ペース駆け|正攻法逃げ",
        "C": r"番手飛|競り|競勝|番手奪",
        "D": r"前不発|目標が不発|目標共倒れ|不発ライン|前が不発",
        "E": r"後方置かれ|最後方|後方|後手ライン",
        "F": r"牽制|進路|詰まり|阻ま|張られ|捌かれ",
        "G": r"位置取|追上|斬り込|脚使",
        "H": r"再仕掛|立て直|仕掛け直",
        "I": r"脚余|仕掛け遅|余し|届かず",
    }
    for k,pat in defs.items(): hist[f"prior_{k}"] = comment.str.contains(pat,regex=True).astype(int)
    # Bは「長く踏んでなお3着内」に限定する。
    hist["prior_B"] = (hist.prior_B.eq(1) & back.eq(1) & finish.le(3)).astype(int)
    hist["prior_finish"] = finish
    hist["prior_back"] = back
    hist["prior_start"] = start
    hist["prior_lap_rel"] = lap - hist.groupby("race_id")["prev_cur_final_lap_time"].transform("mean")
    keep=["race_id","car_no","has_previous_day","prior_finish","prior_back","prior_start","prior_lap_rel","prev_cur_result_comment"]+[f"prior_{x}" for x in defs]
    r=r.merge(hist[keep],on=["race_id","car_no"],how="left")
    for c in PRIOR: r[c]=pd.to_numeric(r[c],errors="coerce").fillna(0)
    r["prior_fact"] = r["prev_cur_result_comment"].fillna("").where(r.has_previous_day.eq(1),"")
    return r


def fit_hgb(df, features, label):
    m=HistGradientBoostingClassifier(learning_rate=.045,max_iter=180,max_leaf_nodes=23,min_samples_leaf=55,l2_regularization=2.0,random_state=SEED)
    m.fit(df[features].fillna(-99),df[label].astype(int)); return m


def metric(y,p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7); y=np.asarray(y,int)
    return {"rows":int(len(y)),"positive_rate":float(y.mean()),"auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,"brier":float(brier_score_loss(y,p)),"logloss":float(log_loss(y,p))}


def race_event_table(df):
    def one(g):
        c=" ".join(g.result_comment.fillna("").astype(str))
        h=g.loc[g.actual_start.eq(1),"car_no"].tolist(); b=g.loc[g.actual_back.eq(1),"car_no"].tolist()
        change=int(bool(h and b and h[0]!=b[0]))
        explicit=int(bool(re.search(r"先行争|叩き合|踏み合|叩き叩かれ",c)))
        return pd.Series({"early_battle_label":int(change or explicit),"explicit_battle_label":explicit})
    labels=df.groupby("race_id",sort=False).apply(one,include_groups=False).reset_index()
    leader=df[df.is_leader.eq(1)]
    agg=df.groupby("race_id",sort=False).agg(race_date=("race_date","first"),line_count=("line_count","first"),b_top_gap=("b_top_gap","first"),h_top_gap=("h_top_gap","first"),leader_b_gap=("leader_b_gap","first"),leader_h_gap=("leader_h_gap","first"),escape_leader_count=("escape_leader_count","first"),max_leader_b=("b_count","max"),max_leader_h=("h_count","max")).reset_index()
    return agg.merge(labels,on="race_id")


EVENT_FEATURES=["line_count","b_top_gap","h_top_gap","leader_b_gap","leader_h_gap","escape_leader_count","max_leader_b","max_leader_h"]


def venue_tables(component):
    base={"back_win":component.loc[component.actual_back.eq(1),"y_win"].mean(),"back_top2":component.loc[component.actual_back.eq(1),"y_top2"].mean(),"bandte_win":component.loc[component.is_bandte.eq(1),"y_win"].mean(),"third_top3":component.loc[component.is_third.eq(1),"y_top3"].mean(),"makuri_win":component.loc[component.winning_move.eq("捲"),"y_win"].mean()}
    rows=[]
    for v,g in component.groupby("venue_code"):
        def shr(mask,label,overall,k=120):
            x=g.loc[mask,label]; return float((x.sum()+k*overall)/(len(x)+k))
        rows.append({"venue_code":v,
            "back_win":shr(g.actual_back.eq(1),"y_win",base["back_win"]),
            "back_top2":shr(g.actual_back.eq(1),"y_top2",base["back_top2"]),
            "bandte_win":shr(g.is_bandte.eq(1),"y_win",base["bandte_win"]),
            "third_top3":shr(g.is_third.eq(1),"y_top3",base["third_top3"]),
            "makuri_win":shr(g.winning_move.eq("捲"),"y_win",base["makuri_win"]),})
    tab=pd.DataFrame(rows).set_index("venue_code")
    for c,val in base.items(): tab[c+"_adj"]=np.log(np.clip(tab[c],1e-4,1)/max(val,1e-4))
    return tab.reset_index(),base


def category_effects(component):
    rows=[]; overall={x:component[x].mean() for x in ["y_win","y_top2","y_top3"]}
    for k in list("ABCDEFGHI"):
        mask=component.has_previous_day.eq(1)&component[f"prior_{k}"].eq(1); n=int(mask.sum())
        row={"category":k,"n":n}
        for y in ["y_win","y_top2","y_top3"]:
            rate=float(component.loc[mask,y].mean()) if n else None; row[y+"_rate"]=rate; row[y+"_lift_pp"]=(rate-overall[y])*100 if rate is not None else None
        # 基準モデルに入れるのは十分な母数かつ2連対率が通常より上の分類だけ。
        row["used_in_model"]=bool(n>=80 and row["y_top2_rate"]>overall["y_top2"])
        rows.append(row)
    return pd.DataFrame(rows),overall


def add_component_predictions(frames, component, cat_effect):
    models={}
    for y,raw,total in [("y_back","p_back_raw",1),("y_win","p_win_raw",1),("y_top2","p_top2_raw",2),("y_top3","p_top3_raw",3)]:
        m=fit_hgb(component,FEATURES,y); models[y]=m
        for f in frames: f[raw]=m.predict_proba(f[FEATURES].fillna(-99))[:,1]
    used=cat_effect.loc[cat_effect.used_in_model,"category"].tolist()
    for f in frames:
        # 初日はprior列が0。2日目以降だけ、学習期間で実測有効だった分類を小さく反映。
        bonus=np.zeros(len(f))
        for k in used:
            lift=float(cat_effect.loc[cat_effect.category.eq(k),"y_top2_lift_pp"].iloc[0])/100
            bonus += f[f"prior_{k}"].to_numpy()*np.clip(lift,-.08,.08)
        f["p_win_raw"]=sigmoid(safe_logit(f.p_win_raw)+bonus)
        f["p_top2_raw"]=sigmoid(safe_logit(f.p_top2_raw)+bonus)
        f["p_top3_raw"]=sigmoid(safe_logit(f.p_top3_raw)+bonus)
        candidate=f.is_self_power.eq(1)
        f["p_back_candidate"]=np.where(candidate,f.p_back_raw,0)
        normalize_by_race(f,"p_back_candidate","p_back",1)
        normalize_by_race(f,"p_win_raw","p_win",1); normalize_by_race(f,"p_top2_raw","p_top2",2); normalize_by_race(f,"p_top3_raw","p_top3",3)
    return models,used


def scenario_pair_rows(riders, battle_map, venue):
    venue_idx=venue.set_index("venue_code")
    all_rows=[]
    for race_id,g in riders.groupby("race_id",sort=False):
        g=g.sort_values("car_no").copy(); q=float(battle_map.get(race_id,.0)); cars=g.car_no.astype(int).to_numpy(); n=len(g)
        for _,back in g[g.p_back.gt(0)].iterrows():
            w=float(back.p_back); bcar=int(back.car_no); bline=back.line_no
            first=g.loc[g.index.repeat(n)].reset_index(drop=True)
            second=pd.concat([g.reset_index(drop=True)]*n,ignore_index=True)
            p=pd.DataFrame({"race_id":race_id,"scenario_back_car":bcar,"scenario_weight":w,"battle_probability":q,
                "first_car":first.car_no.astype(int),"second_car":second.car_no.astype(int)})
            mask=p.first_car.ne(p.second_car); p=p[mask].copy(); first=first[mask].reset_index(drop=True); second=second[mask].reset_index(drop=True); p=p.reset_index(drop=True)
            p["log_first_win"]=np.log(np.clip(first.p_win.to_numpy(),1e-8,1)); p["log_second_top2"]=np.log(np.clip(second.p_top2.to_numpy()/2,1e-8,1))
            p["first_is_back"]=(p.first_car==bcar).astype(int); p["second_is_back"]=(p.second_car==bcar).astype(int)
            fb=(first.line_no.eq(bline)&first.line_position.eq(2)); sb=(second.line_no.eq(bline)&second.line_position.eq(2))
            ft=(first.line_no.eq(bline)&first.line_position.eq(3)); st=(second.line_no.eq(bline)&second.line_position.eq(3))
            p["first_is_bandte_of_back"]=fb.astype(int).to_numpy(); p["second_is_bandte_of_back"]=sb.astype(int).to_numpy(); p["first_is_third_of_back"]=ft.astype(int).to_numpy(); p["second_is_third_of_back"]=st.astype(int).to_numpy()
            p["first_other_leader"]=(first.is_leader.eq(1)&first.line_no.ne(bline)).astype(int).to_numpy(); p["second_other_leader"]=(second.is_leader.eq(1)&second.line_no.ne(bline)).astype(int).to_numpy()
            p["same_line"]=first.line_no.eq(second.line_no).astype(int).to_numpy(); p["front_bandte"]=(p.first_is_back.eq(1)&p.second_is_bandte_of_back.eq(1)).astype(int); p["bandte_front"]=(p.first_is_bandte_of_back.eq(1)&p.second_is_back.eq(1)).astype(int); p["front_third"]=(p.first_is_back.eq(1)&p.second_is_third_of_back.eq(1)).astype(int); p["third_front"]=(p.first_is_third_of_back.eq(1)&p.second_is_back.eq(1)).astype(int)
            p["line_order"]=(p.same_line.eq(1)&(second.line_position.to_numpy()==first.line_position.to_numpy()+1)).astype(int)
            p["battle_first_makuri"]=q*p.first_other_leader*first.makuri_rel.clip(lower=0).to_numpy(); p["battle_second_front"]=q*p.second_is_back; p["battle_second_bandte"]=q*p.second_is_bandte_of_back
            p["score_diff"]=first.score.to_numpy()-second.score.to_numpy(); p["first_line_position"]=first.line_position.to_numpy(); p["second_line_position"]=second.line_position.to_numpy()
            try: va=venue_idx.loc[str(g.venue_code.iloc[0])]
            except KeyError:
                try: va=venue_idx.loc[g.venue_code.iloc[0]]
                except KeyError: va=pd.Series({"back_win_adj":0,"bandte_win_adj":0,"third_top3_adj":0,"makuri_win_adj":0})
            p["venue_first_adj"]=p.first_is_back*va.back_win_adj+p.first_is_bandte_of_back*va.bandte_win_adj+p.first_is_third_of_back*va.third_top3_adj+p.first_other_leader*va.makuri_win_adj
            p["venue_second_adj"]=p.second_is_back*va.back_win_adj+p.second_is_bandte_of_back*va.bandte_win_adj+p.second_is_third_of_back*va.third_top3_adj+p.second_other_leader*va.makuri_win_adj
            all_rows.append(p)
    return pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()


def fit_pair_model(pair_calib, results):
    # 同着で同順位が複数ある場合も停止せず、公式2車単との照合は後段で行う。
    hit=results.loc[results.finish_order.isin([1,2])].pivot_table(index="race_id",columns="finish_order",values="car_no",aggfunc="first").rename(columns={1:"winner",2:"second"}).reset_index()
    d=pair_calib.merge(hit,on="race_id",how="left"); d["is_hit"]=(d.first_car.eq(d.winner)&d.second_car.eq(d.second)).astype(int)
    # scenario rows are mixtures; actual B scenario carries the supervised relation. Others remain weighted alternatives.
    actual=results.loc[results.actual_back.eq(1),["race_id","car_no"]].rename(columns={"car_no":"actual_back_car"})
    d=d.merge(actual,on="race_id",how="left"); d=d[d.scenario_back_car.eq(d.actual_back_car)].copy()
    m=LogisticRegression(C=.25,max_iter=500,class_weight="balanced",random_state=SEED)
    m.fit(d[PAIR_FEATURES].fillna(0),d.is_hit); return m


def predict_pairs(scenarios, model):
    d=scenarios.copy(); raw=model.predict_proba(d[PAIR_FEATURES].fillna(0))[:,1]; d["raw"]=raw
    den=d.groupby(["race_id","scenario_back_car"]).raw.transform("sum"); d["scenario_pair_probability"]=d.raw/den
    d["weighted_probability"]=d.scenario_weight*d.scenario_pair_probability
    out=d.groupby(["race_id","first_car","second_car"],as_index=False).agg(pair_probability=("weighted_probability","sum"))
    den2=out.groupby("race_id").pair_probability.transform("sum"); out.pair_probability=out.pair_probability/den2
    return out


PROB_BINS=[0,.01,.02,.05,.10,1.01]
ODDS_BINS=[0,10,20,50,100,10000]


def calibrate_pair_prob(valid, test):
    iso=IsotonicRegression(out_of_bounds="clip",y_min=1e-6,y_max=.95).fit(valid.pair_probability,valid.is_hit)
    for d in [valid,test]:
        d["cal_raw"]=iso.predict(d.pair_probability); den=d.groupby("race_id").cal_raw.transform("sum"); d["calibrated_probability"]=d.cal_raw/den
    return iso


def reliability_tables(valid):
    v=valid.copy(); v["prob_band"]=pd.cut(v.calibrated_probability,PROB_BINS,right=False,include_lowest=True); v["odds_band"]=pd.cut(v.exacta_odds,ODDS_BINS,right=False,include_lowest=True)
    prob=v.groupby("prob_band",observed=False).agg(n=("is_hit","size"),pred=("calibrated_probability","mean"),actual=("is_hit","mean")).reset_index(); prob["ratio"]=prob.actual/prob.pred
    odds=v.groupby("odds_band",observed=False).agg(n=("is_hit","size"),pred=("calibrated_probability","mean"),actual=("is_hit","mean")).reset_index(); odds["raw_ratio"]=odds.actual/odds.pred
    odds["reliability_factor"]=(odds.n*odds.raw_ratio+500*1)/(odds.n+500); odds["reliability_factor"]=odds.reliability_factor.clip(.25,1.20)
    # 同じ予測確率でも高オッズ帯だけ外れる現象を検証期間で補正する。
    # 予測確率そのものはオッズを見る前に確定済みで、ここは購入用EVの安全係数だけ。
    joint=v.groupby(["prob_band","odds_band"],observed=False).agg(n=("is_hit","size"),pred=("calibrated_probability","mean"),actual=("is_hit","mean")).reset_index()
    joint["raw_ratio"]=joint.actual/joint.pred
    om={str(k):v for k,v in zip(odds.odds_band,odds.reliability_factor)}
    joint["odds_prior"]=joint.odds_band.astype(str).map(om).fillna(1.0)
    joint["reliability_factor"]=(joint.n*joint.raw_ratio+300*joint.odds_prior)/(joint.n+300)
    joint["reliability_factor"]=joint.reliability_factor.replace([np.inf,-np.inf],np.nan).fillna(joint.odds_prior).clip(.10,1.20)
    return prob,odds,joint


def attach_market(pairs, odds, races, results):
    d=pairs.merge(odds,on=["race_id","first_car","second_car"],how="inner")
    d=d.merge(races[["race_id","race_date","day_no","venue_code","venue_name","race_class","actual_starters","two_car_exacta_combination","two_car_exacta_payout_yen"]],on="race_id",how="left")
    d["combination"]=d.first_car.astype(int).astype(str)+"-"+d.second_car.astype(int).astype(str); d["is_hit"]=d.combination.eq(d.two_car_exacta_combination).astype(int)
    return d


def apply_reliability(d, joint_table):
    x=d.copy(); x["odds_band"]=pd.cut(x.exacta_odds,ODDS_BINS,right=False,include_lowest=True); x["prob_band"]=pd.cut(x.calibrated_probability,PROB_BINS,right=False,include_lowest=True)
    mp={f"{p}|{o}":v for p,o,v in zip(joint_table.prob_band.astype(str),joint_table.odds_band.astype(str),joint_table.reliability_factor)}
    key=x.prob_band.astype(str)+"|"+x.odds_band.astype(str)
    x["odds_reliability"]=key.map(mp).fillna(.50); x["purchase_probability"]=x.calibrated_probability*x.odds_reliability; x["ev"]=x.purchase_probability*x.exacta_odds
    entropy_part=-(x.calibrated_probability*np.log(np.clip(x.calibrated_probability,1e-12,1)))
    x["race_entropy"]=entropy_part.groupby(x.race_id).transform("sum")/np.log(x.groupby("race_id").race_id.transform("size"))
    return x


def portfolio(d,min_ev,min_prob,max_points,confidence_max=1.0):
    q=d[(d.ev>=min_ev)&(d.purchase_probability>=min_prob)&(d.race_entropy<=confidence_max)].copy()
    q=q.sort_values(["race_id","ev","calibrated_probability"],ascending=[True,False,False])
    q=q[q.groupby("race_id").cumcount()<max_points]
    if q.empty: return {"purchase_races":0,"bets":0,"roi":0,"max_points":max_points,"min_ev":min_ev,"min_prob":min_prob,"confidence_max":confidence_max},pd.DataFrame()
    b=q.sort_values(["race_date","race_id","ev"],ascending=[True,True,False]).copy(); b["stake"]=100; b["return"]=np.where(b.is_hit.eq(1),pd.to_numeric(b.two_car_exacta_payout_yen,errors="coerce").fillna(0),0)
    rp=b.groupby(["race_date","race_id"],sort=True).apply(lambda x:float(x["return"].sum()-x.stake.sum()),include_groups=False); eq=rp.cumsum(); dd=eq.cummax()-eq
    streak=mx=0
    for z in (rp<0): streak=streak+1 if z else 0; mx=max(mx,streak)
    stake=float(b.stake.sum()); ret=float(b["return"].sum())
    return {"purchase_races":int(b.race_id.nunique()),"bets":int(len(b)),"avg_points":float(len(b)/b.race_id.nunique()),"hits":int(b.is_hit.sum()),"bet_hit_rate":float(b.is_hit.mean()),"race_hit_rate":float(b.groupby("race_id").is_hit.max().mean()),"stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,"roi":ret/stake if stake else 0,"max_losing_streak":int(mx),"max_drawdown_yen":float(dd.max() if len(dd) else 0),"max_points":max_points,"min_ev":min_ev,"min_prob":min_prob,"confidence_max":confidence_max},b


def choose_rules(valid,total_races):
    rows=[]
    for pts in range(1,6):
        for ev in [1.05,1.10,1.20,1.30,1.50,1.80,2.00]:
            for pr in [.01,.02,.03,.05]:
                for conf in [.70,.80,.90,1.0]:
                    m,_=portfolio(valid,ev,pr,pts,conf); m["purchase_rate"]=m.get("purchase_races",0)/total_races; rows.append(m)
    z=pd.DataFrame(rows); eligible=z[(z.purchase_races>=100)&(z.purchase_rate<=.35)].copy()
    chosen=[]
    for pts in range(1,6):
        q=eligible[eligible.max_points.eq(pts)]
        if q.empty: q=z[z.max_points.eq(pts)]
        chosen.append(q.sort_values(["roi","purchase_races"],ascending=[False,False]).iloc[0].to_dict())
    return z,pd.DataFrame(chosen)


def breakdown(bets,races):
    if bets.empty:return pd.DataFrame()
    b=bets.copy(); b["day_group"]=np.select([b.day_no.eq(1),b.day_no.eq(2)],["初日","2日目"],default="最終日等"); b["class_group"]=np.where(b.race_class.astype(str).str.contains("Ｓ|S"),"S級",np.where(b.race_class.astype(str).str.contains("Ａ|A"),"A級","その他")); b["starters_group"]=b.actual_starters.astype("Int64").astype(str)+"車"
    rows=[]
    for col in ["day_group","class_group","starters_group","venue_name"]:
        for key,g in b.groupby(col):
            stake=100*len(g); ret=float(g["return"].sum()); rows.append({"dimension":col,"segment":key,"purchase_races":int(g.race_id.nunique()),"bets":len(g),"hits":int(g.is_hit.sum()),"stake_yen":stake,"return_yen":ret,"roi":ret/stake if stake else 0})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-dir",default="dataset"); ap.add_argument("--output-dir",default="chatgpt_baseline"); a=ap.parse_args(); src=Path(a.dataset_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    races=pd.read_csv(src/"races_2025.csv",dtype={"race_id":str,"venue_code":str}); riders=pd.read_csv(src/"rider_features_2025.csv",dtype={"race_id":str,"venue_code":str}); results=pd.read_csv(src/"official_results_2025.csv",dtype={"race_id":str}); odds=pd.read_csv(src/"exacta_odds_2025.csv",dtype={"race_id":str})
    for d in [riders,results,odds,races]:
        for c in ["car_no","first_car","second_car","finish_order","actual_back","actual_start","exacta_odds","race_date"]:
            if c in d:d[c]=pd.to_numeric(d[c],errors="coerce")
    df=add_prerace_features(riders,races,results)
    lab=results[["race_id","car_no","finish_order","actual_back","actual_start","winning_move","final_lap_time","result_comment"]]
    df=df.merge(lab,on=["race_id","car_no"],how="inner"); df=df[df.finish_order.gt(0)].copy(); df["y_win"]=df.finish_order.eq(1).astype(int); df["y_top2"]=df.finish_order.le(2).astype(int); df["y_top3"]=df.finish_order.le(3).astype(int); df["y_back"]=df.actual_back.fillna(0).astype(int)
    component=df[df.race_date<=COMPONENT_END].copy(); pair_calib=df[(df.race_date>COMPONENT_END)&(df.race_date<=PAIR_END)].copy(); valid=df[(df.race_date>PAIR_END)&(df.race_date<=VALID_END)].copy(); test=df[df.race_date>=TEST_START].copy()
    cats,overall=category_effects(component); models,used=add_component_predictions([component,pair_calib,valid,test],component,cats)
    events=race_event_table(df); event_train=events[events.race_date<=COMPONENT_END]; event_model=LogisticRegression(max_iter=500,class_weight="balanced",random_state=SEED).fit(event_train[EVENT_FEATURES].fillna(0),event_train.early_battle_label)
    battle_metrics=[]; battle_maps={}
    for name,f in [("pair",pair_calib),("validation",valid),("test",test)]:
        e=events[events.race_id.isin(f.race_id.unique())].copy(); e["p_battle"]=event_model.predict_proba(e[EVENT_FEATURES].fillna(0))[:,1]; battle_maps[name]=dict(zip(e.race_id,e.p_battle)); battle_metrics.append({"split":name,**metric(e.early_battle_label,e.p_battle),"explicit_positive_rate":float(e.explicit_battle_label.mean())})
    venue,venue_base=venue_tables(component)
    sc_pair=scenario_pair_rows(pair_calib,battle_maps["pair"],venue); pair_model=fit_pair_model(sc_pair,results)
    pv=predict_pairs(scenario_pair_rows(valid,battle_maps["validation"],venue),pair_model); pt=predict_pairs(scenario_pair_rows(test,battle_maps["test"],venue),pair_model)
    valid_market=attach_market(pv,odds,races,results); test_market=attach_market(pt,odds,races,results)
    calibrate_pair_prob(valid_market,test_market); prob_cal,odds_rel,joint_rel=reliability_tables(valid_market); valid_market=apply_reliability(valid_market,joint_rel); test_market=apply_reliability(test_market,joint_rel)
    grid,chosen=choose_rules(valid_market,valid.race_id.nunique())
    comparisons=[]; chosen_bets={}
    for _,rule in chosen.iterrows():
        m,b=portfolio(test_market,float(rule.min_ev),float(rule.min_prob),int(rule.max_points),float(rule.confidence_max)); m["test_races"]=int(test.race_id.nunique()); m["purchase_rate"]=m.get("purchase_races",0)/test.race_id.nunique(); comparisons.append(m); chosen_bets[int(rule.max_points)]=b
    comp=pd.DataFrame(comparisons); selected_rule=chosen.sort_values(["roi","purchase_races"],ascending=[False,False]).iloc[0]; selected_points=int(selected_rule.max_points); final_bets=chosen_bets[selected_points]
    top=[]
    for k in [1,2,3,4,5]: top.append({"k":k,"hit_rate":float(test_market.sort_values(["race_id","calibrated_probability"],ascending=[True,False]).groupby("race_id").head(k).groupby("race_id").is_hit.max().mean())})
    rider_metrics={}
    for y,p in [("y_back","p_back"),("y_win","p_win"),("y_top2","p_top2"),("y_top3","p_top3")]:rider_metrics[y]=metric(test[y],test[p])
    report={"execution":"Python actual run","model":"ChatGPT conversation method baseline","identity_features_used":False,"odds_used_after_probability_only":True,"split":{"component_train":"2025-01-01..2025-06-30","pair_train":"2025-07-01..2025-08-31","validation":"2025-09-01..2025-10-31","locked_test":"2025-11-01..2025-12-31"},"dataset":{"races":int(races.race_id.nunique()),"riders":len(riders),"odds":len(odds),"test_races":int(test.race_id.nunique())},"used_prior_categories":used,"rider_metrics":rider_metrics,"battle_metrics":battle_metrics,"topk_exacta":top,"locked_rules_by_max_points":chosen.to_dict("records"),"locked_test_purchase_comparison":comparisons,"selected_baseline_max_points":selected_points,"selected_baseline_result":next(x for x in comparisons if x["max_points"]==selected_points),"notes":["2車単は各バック取得シナリオの条件付き順序確率を混合。1着率×2着率の単純積ではない。","前日短評が取得できない分類は0ではなく未検出。カテゴリ母数と根拠を別CSVに保存。","場補正は学習期間の場別バック残り・番手・3番手・捲り実測を縮約して使用。","オッズ帯信頼係数は予測確率完成後、検証期間だけで算定。"]}
    (out/"baseline_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    cats.to_csv(out/"lose_strong_category_effects.csv",index=False,encoding="utf-8-sig"); venue.to_csv(out/"venue_corrections.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(battle_metrics).to_csv(out/"battle_metrics.csv",index=False,encoding="utf-8-sig"); prob_cal.to_csv(out/"probability_calibration.csv",index=False,encoding="utf-8-sig"); odds_rel.to_csv(out/"odds_reliability.csv",index=False,encoding="utf-8-sig"); joint_rel.to_csv(out/"probability_odds_joint_reliability.csv",index=False,encoding="utf-8-sig"); grid.to_csv(out/"validation_rule_grid.csv",index=False,encoding="utf-8-sig"); chosen.to_csv(out/"locked_rules.csv",index=False,encoding="utf-8-sig"); comp.to_csv(out/"test_purchase_comparison.csv",index=False,encoding="utf-8-sig"); breakdown(final_bets,races).to_csv(out/"test_breakdowns.csv",index=False,encoding="utf-8-sig")
    test[["race_id","race_date","venue_name","race_no","car_no","line_no","line_position","p_back","p_win","p_top2","p_top3","prior_fact"]+[f"prior_{x}" for x in "ABCDEFGHI"]+["finish_order"]].to_csv(out/"test_rider_predictions.csv",index=False,encoding="utf-8-sig")
    test_market[["race_id","first_car","second_car","pair_probability","calibrated_probability","exacta_odds","odds_reliability","purchase_probability","ev","is_hit"]].to_csv(out/"test_pair_predictions.csv",index=False,encoding="utf-8-sig")
    final_bets.to_csv(out/"selected_test_bets.csv",index=False,encoding="utf-8-sig")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
