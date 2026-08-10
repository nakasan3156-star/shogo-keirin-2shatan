#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import venue_edge_analysis as v


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-dir',required=True);ap.add_argument('--external-dir',required=True);ap.add_argument('--output-dir',default='venue_third_exact');a=ap.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    r25,u25,y25,o25=v.load(Path(a.train_dir),'2025')
    r26,u26,y26,o26=v.load(Path(a.external_dir),'2026_01_02')
    r26=r26[r26.race_date.between(v.EXT_START,v.EXT_END)].copy();ids=set(r26.race_id);u26=u26[u26.race_id.isin(ids)];y26=y26[y26.race_id.isin(ids)]
    d=v.prepare_rider_frame(r25,u25,y25,r26,u26,y26)
    f=d[d.is_back_third].copy().reset_index(drop=True)
    f['y']=f.finish_order.eq(3).astype(int)
    f=f[f.race_date.between(20250101,v.EXT_END)].copy()
    disc=f[f.race_date.le(v.DISC_END)].copy()
    pred=v.discovery_oof(disc,v.RIDER_MODEL_FEATURES)
    model=v.fit_hgb(disc,v.RIDER_MODEL_FEATURES)
    f['expected']=model.predict_proba(f[v.RIDER_MODEL_FEATURES].fillna(-99))[:,1]
    mp={(r,c):p for r,c,p in zip(disc.race_id,disc.car_no,pred)};mask=f.race_date.le(v.DISC_END)
    f.loc[mask,'expected']=[mp.get((r,c),e) for r,c,e in zip(f.loc[mask,'race_id'],f.loc[mask,'car_no'],f.loc[mask,'expected'])]
    f['split']=f.race_date.astype(int).map(v.split_name);f['resid']=f.y-f.expected
    rows=[]
    for (split,vc,vn),g in f.groupby(['split','venue_code','venue_name']):
        if split=='other':continue
        n=len(g);res=float(g.resid.mean());se=float(g.resid.std(ddof=1)/math.sqrt(n)) if n>1 else np.nan
        rows.append({'venue_code':str(vc).zfill(2),'venue_name':vn,'split':split,'n':n,'actual_exact_third_rate':float(g.y.mean()),'expected_rate':float(g.expected.mean()),'residual_pp':res*100,'ci95_low_pp':(res-1.96*se)*100 if np.isfinite(se) else np.nan,'ci95_high_pp':(res+1.96*se)*100 if np.isfinite(se) else np.nan})
    agg=pd.DataFrame(rows);agg.to_csv(out/'backline_third_exact_by_split.csv',index=False,encoding='utf-8-sig')
    pv=agg.pivot_table(index=['venue_code','venue_name'],columns='split',values=['n','actual_exact_third_rate','residual_pp'],aggfunc='first').reset_index();pv.columns=['_'.join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in pv.columns]
    def val(r,key,default=np.nan):
        z=r.get(key,default);return float(z) if pd.notna(z) else default
    rr=[]
    for _,r in pv.iterrows():
        nd=val(r,'n_discovery_2025_01_06',0);nv=val(r,'n_validation_2025_09_10',0);nt=val(r,'n_locked_test_2025_11_12',0);ne=val(r,'n_external_2026_01_02',0)
        rd=val(r,'residual_pp_discovery_2025_01_06');rv=val(r,'residual_pp_validation_2025_09_10');rt=val(r,'residual_pp_locked_test_2025_11_12');re=val(r,'residual_pp_external_2026_01_02')
        candidate=nd>=v.MIN_DISC_N and nv>=v.MIN_VALID_N and np.isfinite(rd) and np.isfinite(rv) and rd*rv>0 and abs(rd)>=.75 and abs(rv)>=.50
        true=bool(candidate and nt>=v.MIN_TEST_N and ne>=v.MIN_EXT_N and np.isfinite(rt) and np.isfinite(re) and rd*rt>0 and rd*re>0)
        if true:cl='本当に再現した癖'
        elif candidate:cl='弱い傾向'
        elif nd>=v.MIN_DISC_N and np.isfinite(rd) and abs(rd)>=1.0:cl='再現しなかった一般論'
        else:cl='補正根拠なし'
        rr.append({'venue_code':r['venue_code'],'venue_name':r['venue_name'],'metric':'backline_third_exact','metric_ja':'B取得ライン3番手・3着ちょうど率','classification':cl,'direction':'↑' if np.isfinite(rd) and rd>0 else '↓' if np.isfinite(rd) else '', 'discovery_n':int(nd),'validation_n':int(nv),'test_n':int(nt),'external_n':int(ne),'discovery_residual_pp':rd,'validation_residual_pp':rv,'test_residual_pp':rt,'external_residual_pp':re,'external_exact_third_rate':val(r,'actual_exact_third_rate_external_2026_01_02')})
    res=pd.DataFrame(rr).sort_values('venue_code');res.to_csv(out/'backline_third_exact_classification.csv',index=False,encoding='utf-8-sig')
    print(res.to_csv(index=False))

if __name__=='__main__':main()
