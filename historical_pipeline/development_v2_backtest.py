#!/usr/bin/env python3
"""展開精度と長期安定性を優先したPR31比較モデル。

開発・温度補正・購入条件の決定は2025年だけ。指定された外部期間は
PR31固定条件とV2固定条件の最終比較にしか使用しない。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import chatgpt_baseline_backtest as base

PR31_RULE = {"max_points": 5, "min_ev": 2.0, "min_prob": 0.03, "confidence_max": 1.0}


def load(root: Path, label: str):
    races=pd.read_csv(root/f"races_{label}.csv",dtype={"race_id":str,"venue_code":str})
    riders=pd.read_csv(root/f"rider_features_{label}.csv",dtype={"race_id":str,"venue_code":str})
    results=pd.read_csv(root/f"official_results_{label}.csv",dtype={"race_id":str})
    odds=pd.read_csv(root/f"exacta_odds_{label}.csv",dtype={"race_id":str})
    for d in [races,riders,results,odds]:
        for c in ["race_date","car_no","first_car","second_car","finish_order","actual_back","actual_start","exacta_odds"]:
            if c in d:d[c]=pd.to_numeric(d[c],errors="coerce")
    return races,riders,results,odds


def event_v2(df):
    def one(g):
        comments=" ".join(g.result_comment.fillna("").astype(str))
        h=g.loc[g.actual_start.eq(1),"car_no"].tolist(); b=g.loc[g.actual_back.eq(1),"car_no"].tolist()
        change=int(bool(h and b and h[0]!=b[0]))
        explicit=int(bool(re.search(r"先行争|叩き合|踏み合|叩き叩かれ|もがき合",comments)))
        winner=g.loc[g.finish_order.eq(1),"winning_move"].astype(str).tolist()
        makuri_win=int(bool(winner and winner[0]=="捲"))
        # 単なるH/B交代ではなく、踏み合い明記または交代後に捲り決着した消耗戦。
        damaging=int(bool(explicit or (change and makuri_win)))
        return pd.Series({"change_label":change,"explicit_label":explicit,"damaging_battle_label":damaging})
    labels=df.groupby("race_id",sort=False).apply(one,include_groups=False).reset_index()
    agg=df.groupby("race_id",sort=False).agg(
        race_date=("race_date","first"),line_count=("line_count","first"),b_top_gap=("b_top_gap","first"),
        h_top_gap=("h_top_gap","first"),leader_b_gap=("leader_b_gap","first"),leader_h_gap=("leader_h_gap","first"),
        escape_leader_count=("escape_leader_count","first"),max_leader_b=("b_count","max"),max_leader_h=("h_count","max")
    ).reset_index()
    return agg.merge(labels,on="race_id")


def fit_battle_model(train,valid):
    x=base.EVENT_FEATURES; y="damaging_battle_label"
    logistic=LogisticRegression(max_iter=700,class_weight="balanced",random_state=base.SEED).fit(train[x].fillna(0),train[y])
    hgb=HistGradientBoostingClassifier(learning_rate=.04,max_iter=180,max_leaf_nodes=19,min_samples_leaf=60,l2_regularization=3,random_state=base.SEED)
    pos=train[y].mean(); weights=np.where(train[y].eq(1),.5/max(pos,1e-6),.5/max(1-pos,1e-6))
    hgb.fit(train[x].fillna(0),train[y],sample_weight=weights)
    candidates=[]
    for name,model in [("logistic",logistic),("hgb",hgb)]:
        p=model.predict_proba(valid[x].fillna(0))[:,1]; m=base.metric(valid[y],p); candidates.append((m["brier"],name,model,m))
    return sorted(candidates,key=lambda z:z[0])[0],candidates


def temperature_back(frames,pair_train):
    rows=[]
    for temp in [.50,.65,.80,1.0,1.25,1.50,2.0]:
        p=np.clip(pair_train.p_back.to_numpy(float),1e-9,1)**(1/temp)
        z=pd.Series(p,index=pair_train.index).groupby(pair_train.race_id).transform("sum").to_numpy()
        q=p/z
        actual=pair_train.y_back.to_numpy(int).astype(bool)
        loss=float(-np.log(np.clip(q[actual],1e-12,1)).mean())
        rows.append({"temperature":temp,"actual_back_logloss":loss})
    chosen=min(rows,key=lambda x:x["actual_back_logloss"])["temperature"]
    for f in frames:
        p=np.clip(f.p_back.to_numpy(float),1e-9,1)**(1/chosen)
        den=pd.Series(p,index=f.index).groupby(f.race_id).transform("sum").to_numpy();f["p_back"]=p/den
    return chosen,pd.DataFrame(rows)


def add_confidence(market,riders,battle_map):
    x=market.copy()
    back=riders.groupby("race_id").p_back.apply(lambda s:np.sort(s.to_numpy())[::-1]).to_dict()
    win=riders.groupby("race_id").p_win.apply(lambda s:np.sort(s.to_numpy())[::-1]).to_dict()
    def gap(mp,rid):
        z=mp.get(rid,np.array([]));return float(z[0]-z[1]) if len(z)>1 else 0.0
    x["back_margin"]=x.race_id.map(lambda rid:gap(back,rid));x["win_margin"]=x.race_id.map(lambda rid:gap(win,rid))
    x["battle_probability"]=x.race_id.map(battle_map).fillna(.5)
    x["battle_certainty"]=(x.battle_probability-.5).abs()*2
    ordered=x.sort_values(["race_id","calibrated_probability"],ascending=[True,False])
    first=ordered.groupby("race_id").calibrated_probability.first();second=ordered.groupby("race_id").calibrated_probability.nth(1)
    x["pair_margin"]=x.race_id.map((first-second).fillna(0))
    return x


def select_v2(d,rule):
    q=d[(d.ev>=rule["min_ev"])&(d.purchase_probability>=rule["min_prob"])&(d.race_entropy<=rule["confidence_max"])
        &(d.exacta_odds<=rule["max_odds"])&(d.back_margin>=rule["min_back_margin"])&(d.pair_margin>=rule["min_pair_margin"])].copy()
    q=q.sort_values(["race_id","ev","calibrated_probability"],ascending=[True,False,False])
    q=q[q.groupby("race_id").cumcount()<rule["max_points"]]
    if q.empty:return q
    q=q.sort_values(["race_date","race_id","ev"],ascending=[True,True,False]);q["stake"]=100
    q["return"]=np.where(q.is_hit.eq(1),pd.to_numeric(q.two_car_exacta_payout_yen,errors="coerce").fillna(0),0)
    return q


def metrics(b,total_races):
    if b.empty:return {"target_races":int(total_races),"purchase_races":0,"purchase_rate":0,"bets":0,"avg_points":0,"hits":0,"race_hit_rate":0,"roi":0,"max_losing_streak":0,"max_drawdown_yen":0,"odds_20pct_worse_roi":0,"largest_hit_removed_roi":0}
    rp=b.groupby(["race_date","race_id"],sort=True).apply(lambda x:float(x["return"].sum()-x.stake.sum()),include_groups=False)
    streak=mx=0
    for loss in rp.lt(0):streak=streak+1 if loss else 0;mx=max(mx,streak)
    eq=rp.cumsum().to_numpy();peak=np.maximum.accumulate(np.r_[0.,eq])[1:]
    stake=float(b.stake.sum());ret=float(b["return"].sum());largest=float(b["return"].max())
    return {"target_races":int(total_races),"purchase_races":int(b.race_id.nunique()),"purchase_rate":b.race_id.nunique()/total_races,
        "bets":int(len(b)),"avg_points":len(b)/b.race_id.nunique(),"hits":int(b.is_hit.sum()),"bet_hit_rate":float(b.is_hit.mean()),
        "race_hit_rate":float(b.groupby("race_id").is_hit.max().mean()),"stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,"roi":ret/stake,
        "max_losing_streak":int(mx),"max_drawdown_yen":float(np.max(peak-eq)) if len(eq) else 0.,"odds_20pct_worse_roi":ret*.8/stake,
        "largest_hit_payout_yen":largest,"largest_hit_removed_roi":(ret-largest)/stake,
        "payout_10000plus_removed_roi":float(b["return"].where(b["return"].lt(10000),0).sum()/stake)}


def choose_robust_rule(valid,total_races):
    rows=[]
    months=valid.race_date.astype(int).astype(str).str[:6]
    for pts in [1,2,3]:
      for ev in [1.25,1.50,2.0]:
       for prob in [.03,.05,.08]:
        for odds in [20,50,100]:
         for bgap in [0,.05]:
          for pgap in [0,.005]:
           rule={"max_points":pts,"min_ev":ev,"min_prob":prob,"confidence_max":1.0,"max_odds":odds,"min_back_margin":bgap,"min_pair_margin":pgap}
           b=select_v2(valid,rule);m=metrics(b,total_races)
           monthly=[]
           for month in sorted(months.unique()):
               ids=set(valid.loc[months.eq(month),"race_id"]);monthly.append(metrics(b[b.race_id.isin(ids)],len(ids))["roi"])
           m.update(rule);m["worst_month_roi"]=min(monthly) if monthly else 0
           m["robust_score"]=min(m["roi"],m["odds_20pct_worse_roi"],m["largest_hit_removed_roi"],m["worst_month_roi"])
           rows.append(m)
    grid=pd.DataFrame(rows)
    eligible=grid[(grid.purchase_races>=100)&(grid.purchase_rate.between(.03,.25))&(grid.hits>=8)
        &(grid.max_losing_streak<=60)&(grid.worst_month_roi>=.80)&(grid.largest_hit_removed_roi>=.90)].copy()
    if eligible.empty:
        eligible=grid[(grid.purchase_races>=60)&(grid.max_losing_streak<=80)].copy()
    chosen=eligible.sort_values(["robust_score","race_hit_rate","max_losing_streak"],ascending=[False,False,True]).iloc[0].to_dict()
    keys=["max_points","min_ev","min_prob","confidence_max","max_odds","min_back_margin","min_pair_margin"]
    return {k:(int(chosen[k]) if k in ["max_points","max_odds"] else float(chosen[k])) for k in keys},grid


def pipeline(df,races25,results25,odds25,races_ext,results_ext,odds_ext,external_start,external_end):
    component=df[df.race_date<=base.COMPONENT_END].copy();pair=df[(df.race_date>base.COMPONENT_END)&(df.race_date<=base.PAIR_END)].copy();valid=df[(df.race_date>base.PAIR_END)&(df.race_date<=base.VALID_END)].copy();external=df[df.race_date.between(external_start,external_end)].copy()
    cats,_=base.category_effects(component);base.add_component_predictions([component,pair,valid,external],component,cats)
    temp,temp_grid=temperature_back([component,pair,valid,external],pair)
    venue,_=base.venue_tables(component)
    ev=event_v2(df);train_ev=ev[ev.race_date<=base.COMPONENT_END];valid_ev=ev[(ev.race_date>base.PAIR_END)&(ev.race_date<=base.VALID_END)]
    chosen_battle,candidates=fit_battle_model(train_ev,valid_ev);_,battle_name,battle_model,battle_valid_metric=chosen_battle
    maps={};battle_reports=[]
    for name,frame in [("pair",pair),("validation",valid),("external",external)]:
        e=ev[ev.race_id.isin(frame.race_id.unique())].copy();e["p"]=battle_model.predict_proba(e[base.EVENT_FEATURES].fillna(0))[:,1];maps[name]=dict(zip(e.race_id,e.p));battle_reports.append({"split":name,**base.metric(e.damaging_battle_label,e.p)})
    pair_model=base.fit_pair_model(base.scenario_pair_rows(pair,maps["pair"],venue),results25)
    pv=base.predict_pairs(base.scenario_pair_rows(valid,maps["validation"],venue),pair_model);pe=base.predict_pairs(base.scenario_pair_rows(external,maps["external"],venue),pair_model)
    vm=base.attach_market(pv,odds25,races25,results25);em=base.attach_market(pe,odds_ext,races_ext,results_ext)
    base.calibrate_pair_prob(vm,em);_,_,joint=base.reliability_tables(vm);vm=base.apply_reliability(vm,joint);em=base.apply_reliability(em,joint)
    vm=add_confidence(vm,valid,maps["validation"]);em=add_confidence(em,external,maps["external"])
    rule,grid=choose_robust_rule(vm,valid.race_id.nunique());return external,em,rule,grid,temp,temp_grid,battle_name,battle_valid_metric,battle_reports


def pr31_external(df,races25,results25,odds25,races_ext,results_ext,odds_ext,external_start,external_end):
    component=df[df.race_date<=base.COMPONENT_END].copy();pair=df[(df.race_date>base.COMPONENT_END)&(df.race_date<=base.PAIR_END)].copy();valid=df[(df.race_date>base.PAIR_END)&(df.race_date<=base.VALID_END)].copy();external=df[df.race_date.between(external_start,external_end)].copy()
    cats,_=base.category_effects(component);base.add_component_predictions([component,pair,valid,external],component,cats)
    events=base.race_event_table(df);train=events[events.race_date<=base.COMPONENT_END];model=LogisticRegression(max_iter=500,class_weight="balanced",random_state=base.SEED).fit(train[base.EVENT_FEATURES].fillna(0),train.early_battle_label)
    maps={}
    for name,frame in [("pair",pair),("validation",valid),("external",external)]:
        e=events[events.race_id.isin(frame.race_id.unique())];maps[name]=dict(zip(e.race_id,model.predict_proba(e[base.EVENT_FEATURES].fillna(0))[:,1]))
    venue,_=base.venue_tables(component);pm=base.fit_pair_model(base.scenario_pair_rows(pair,maps["pair"],venue),results25)
    pv=base.predict_pairs(base.scenario_pair_rows(valid,maps["validation"],venue),pm);pe=base.predict_pairs(base.scenario_pair_rows(external,maps["external"],venue),pm)
    vm=base.attach_market(pv,odds25,races25,results25);em=base.attach_market(pe,odds_ext,races_ext,results_ext);base.calibrate_pair_prob(vm,em);_,_,joint=base.reliability_tables(vm);em=base.apply_reliability(em,joint)
    _,bets=base.portfolio(em,**PR31_RULE);return external,em,bets


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--train-dir",required=True);ap.add_argument("--external-dir",required=True);ap.add_argument("--external-label",required=True);ap.add_argument("--external-start",type=int,required=True);ap.add_argument("--external-end",type=int,required=True);ap.add_argument("--output-dir",default="development_v2_results");a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    r25,u25,y25,o25=load(Path(a.train_dir),"2025");re,ue,ye,oe=load(Path(a.external_dir),a.external_label);ids=set(re.loc[re.race_date.between(a.external_start,a.external_end),"race_id"]);re=re[re.race_id.isin(ids)];ue=ue[ue.race_id.isin(ids)];ye=ye[ye.race_id.isin(ids)];oe=oe[oe.race_id.isin(ids)]
    races=pd.concat([r25,re],ignore_index=True,sort=False);riders=pd.concat([u25,ue],ignore_index=True,sort=False);results=pd.concat([y25,ye],ignore_index=True,sort=False)
    df=base.add_prerace_features(riders,races,results);lab=results[["race_id","car_no","finish_order","actual_back","actual_start","winning_move","final_lap_time","result_comment"]];df=df.merge(lab,on=["race_id","car_no"],how="inner");df=df[df.finish_order.gt(0)].copy();df["y_win"]=df.finish_order.eq(1).astype(int);df["y_top2"]=df.finish_order.le(2).astype(int);df["y_top3"]=df.finish_order.le(3).astype(int);df["y_back"]=df.actual_back.fillna(0).astype(int)
    # 分岐ごとにコピーし、PR31側をV2の温度補正等で汚染しない。
    ext_v2,market_v2,rule,grid,temp,temp_grid,battle_name,battle_valid,battle_reports=pipeline(df.copy(),r25,y25,o25,re,ye,oe,a.external_start,a.external_end)
    ext_b,market_b,bets_b=pr31_external(df.copy(),r25,y25,o25,re,ye,oe,a.external_start,a.external_end)
    bets_v2=select_v2(market_v2,rule);total=int(re.race_id.nunique());comparison={"PR31_frozen":metrics(bets_b,total),"development_v2":metrics(bets_v2,total)}
    report={"execution":"Python actual run","external_period":f"{a.external_start}..{a.external_end}","external_data_used_for_tuning":False,"PR31_changed":False,"v2_changes":["消耗を伴う先行争いラベルへ分離","非線形候補を2025年検証Brierで選択","バック確率温度を2025年7〜8月だけで補正","展開確信度と高額依存を含む2025年限定の頑健ルール選択"],"external_dataset":{"races":total,"riders":len(ue),"odds":len(oe),"results":len(ye)},"v2_frozen_rule":rule,"back_temperature":temp,"battle_model":battle_name,"battle_validation":battle_valid,"battle_metrics":battle_reports,"comparison":comparison}
    (out/"development_v2_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");bets_b.to_csv(out/"pr31_bets.csv",index=False,encoding="utf-8-sig");bets_v2.to_csv(out/"v2_bets.csv",index=False,encoding="utf-8-sig");market_v2.to_csv(out/"v2_pair_predictions.csv",index=False,encoding="utf-8-sig");grid.to_csv(out/"v2_2025_rule_grid.csv",index=False,encoding="utf-8-sig");temp_grid.to_csv(out/"back_temperature_grid.csv",index=False,encoding="utf-8-sig");pd.DataFrame(battle_reports).to_csv(out/"battle_metrics.csv",index=False,encoding="utf-8-sig");print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
