from __future__ import annotations

from typing import Any


def _max_drawdown(profits: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for profit in profits:
        equity += profit
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def evaluate_predictions(
    races: list[dict[str, Any]],
    thresholds: list[float],
    stake_per_pair: int = 100,
) -> dict[str, Any]:
    if not races:
        raise ValueError("バックテスト対象レースがありません。")
    results = []
    for threshold in thresholds:
        stake = 0.0
        returned = 0.0
        wins = 0
        profits: list[float] = []
        for race in races:
            actual = (int(race["actual_first"]), int(race["actual_second"]))
            actual_payout = race.get("actual_payout_yen_per_100")
            race_stake = 0.0
            race_return = 0.0
            for pair in race["prediction"]["all_combinations"]:
                if float(pair["ev"]) < threshold:
                    continue
                stake += stake_per_pair
                race_stake += stake_per_pair
                if (int(pair["first"]), int(pair["second"])) == actual:
                    if actual_payout is not None:
                        payout = (float(stake_per_pair) / 100.0) * float(actual_payout)
                    else:
                        payout = float(stake_per_pair) * float(pair["odds"])
                    returned += payout
                    race_return += payout
                    wins += 1
            # 同一レースの全車券は同時決済として扱う。組み合わせの並び順で
            # 最大ドローダウンが変わらないよう、損益はレース単位で記録する。
            profits.append(race_return - race_stake)
        bets = int(stake / stake_per_pair)
        results.append(
            {
                "ev_threshold": threshold,
                "races": len(races),
                "bets": bets,
                "wins": wins,
                "stake": int(stake),
                "return": round(returned, 2),
                "roi": round(returned / stake, 6) if stake else None,
                "hit_rate_per_bet": round(wins / bets, 6) if bets else None,
                "max_drawdown": round(_max_drawdown(profits), 2),
            }
        )
    return {
        "status": "descriptive_backtest_only",
        "warning": "同じレースで閾値を選ぶと過学習になります。調整期間と未使用検証期間を分離してください。",
        "results": results,
    }
