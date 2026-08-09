#!/usr/bin/env python3
"""基準モデル出力へ、個別効果・場補正・DD・オッズ悪化監査を追記する。"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

TEST_START=20251101

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset-dir',default='dataset');ap.add_argument('--result-dir',default='chatgpt_baseline');a=ap.parse_args();src=Path(a.dataset_dir);out=Path(a.result_dir)
    report=json.loads((out/'baseline_report.json').read_text(encoding='utf-8'))
    riders=pd.read_csv(src/'rider_features_2025.csv',dtype={'race_id':str,'venue_code':str});res=pd.read_csv(src/'official_results_2025.csv',dtype={'race_id':str});races=pd.read_csv(src/'races_2025.csv',dtype={'race_id':str,'venue_code':str});bets=pd.read_csv(out/'selected_test_bets.csv',dtype={'race_id':str})
    for d in [riders,res,races]:
        for c in ['race_date','car_no','finish_order','actual_back','actual_start','line_position']:
            if c in d:d[c]=pd.to_numeric(d[c],errors='coerce')
    test=riders[riders.race_date>=TEST_START].merge(res,on=['race_id','car_no'],how='inner',suffixes=('','_result'))
    back=test.loc[test.actual_back.eq(1),['race_id','line_no','car_no']].rename(columns={'line_no':'back_line','car_no':'back_car'})
    t=test.merge(back,on='race_id',how='left');t['is_back_holder']=t.car_no.eq(t.back_car);t['is_back_bandte']=t.line_no.eq(t.back_line)&t.line_position.eq(2);t['is_back_third']=t.line_no.eq(t.back_line)&t.line_position.eq(3);t['is_other_leader']=t.line_position.eq(1)&t.line_no.ne(t.back_line)
    overall={f'finish_le_{k}':float(t.finish_order.le(k).mean()) for k in [1,2,3]}
    rows=[]
    for name,col in [('バック取得者','is_back_holder'),('バック取得ライン番手','is_back_bandte'),('バック取得ライン3番手','is_back_third'),('別線先頭','is_other_leader')]:
        g=t[t[col]];row={'item':name,'n':len(g)}
        for k in [1,2,3]:
            rate=float(g.finish_order.le(k).mean()) if len(g) else None;row[f'top{k}_rate']=rate;row[f'top{k}_lift_pp']=(rate-overall[f'finish_le_{k}'])*100 if rate is not None else None
        rows.append(row)
    winners=t[t.finish_order.eq(1)][['race_id','line_no','winning_move']].rename(columns={'line_no':'winner_line','winning_move':'winner_move'});seconds=t[t.finish_order.eq(2)][['race_id','line_no','car_no']].rename(columns={'line_no':'second_line','car_no':'second_car'});rr=back.merge(winners,on='race_id').merge(seconds,on='race_id')
    cross=rr.winner_line.ne(rr.back_line)&rr.winner_move.eq('捲'); otherwin=rr.winner_line.ne(rr.back_line)
    rows += [
      {'item':'別線捲り1着','n':len(rr),'event_rate':float(cross.mean())},
      {'item':'別線1着時の前残り2着','n':int(otherwin.sum()),'event_rate':float((rr.loc[otherwin,'second_car']==rr.loc[otherwin,'back_car']).mean()) if otherwin.any() else None},
      {'item':'別線1着時のバックライン番手2着','n':int(otherwin.sum()),'event_rate':float((rr.loc[otherwin,'second_line']==rr.loc[otherwin,'back_line']).mean()) if otherwin.any() else None},
    ]
    effects=pd.DataFrame(rows);effects.to_csv(out/'development_effects.csv',index=False,encoding='utf-8-sig')

    # 学習期間で作った場補正の方向がテストでも再現するか（場別率の順位相関）。
    venue=pd.read_csv(out/'venue_corrections.csv',dtype={'venue_code':str});checks=[]
    role_defs=[('back_win_adj',t.is_back_holder,t.finish_order.eq(1)),('bandte_win_adj',t.is_back_bandte,t.finish_order.eq(1)),('third_top3_adj',t.is_back_third,t.finish_order.le(3)),('makuri_win_adj',t.is_other_leader,t.winning_move.eq('捲')&t.finish_order.eq(1))]
    for adj,mask,y in role_defs:
        z=t.loc[mask,['venue_code']].copy();z['y']=y[mask].astype(int).to_numpy();obs=z.groupby('venue_code').agg(n=('y','size'),test_rate=('y','mean')).reset_index().merge(venue[['venue_code',adj]],on='venue_code',how='inner');corr=float(obs.loc[obs.n>=10,[adj,'test_rate']].corr(method='spearman').iloc[0,1]) if (obs.n>=10).sum()>=3 else None;checks.append({'venue_effect':adj,'venues_n10':int((obs.n>=10).sum()),'spearman_train_adjustment_vs_test_rate':corr})
    pd.DataFrame(checks).to_csv(out/'venue_effect_validation.csv',index=False,encoding='utf-8-sig')

    # DDは開始資金0円を過去最高値に含める。
    bets['stake']=100;bets['return']=np.where(bets.is_hit.eq(1),pd.to_numeric(bets.two_car_exacta_payout_yen,errors='coerce').fillna(0),0);rp=bets.sort_values(['race_date','race_id']).groupby(['race_date','race_id'],sort=True).apply(lambda x:float(x['return'].sum()-x.stake.sum()),include_groups=False);eq=rp.cumsum();peak=np.maximum.accumulate(np.r_[0,eq.to_numpy()])[1:];maxdd=float(np.max(peak-eq.to_numpy())) if len(eq) else 0
    selected=report['selected_baseline_result'];selected['max_drawdown_yen']=maxdd
    selected['odds_deterioration_roi']={f'minus_{x}%':float((bets['return']*(1-x/100)).sum()/bets.stake.sum()) for x in [5,10,15,20]}
    report['selected_baseline_result']=selected;report['development_effects_file']='development_effects.csv';report['venue_effect_validation_file']='venue_effect_validation.csv';report['audit_finalized_by_python']=True
    (out/'baseline_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selected':selected,'development_effects':rows,'venue_checks':checks},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
