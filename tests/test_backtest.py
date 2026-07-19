from app.backtest import evaluate_predictions


def test_backtest_flat_stake_math():
    prediction = {
        "all_combinations": [
            {"first": 1, "second": 2, "ev": 1.2, "odds": 3.0},
            {"first": 2, "second": 1, "ev": 0.9, "odds": 4.0},
        ]
    }
    output = evaluate_predictions(
        [{"prediction": prediction, "actual_first": 1, "actual_second": 2, "actual_payout_yen_per_100": 250}],
        [1.0],
        100,
    )
    result = output["results"][0]
    assert result["bets"] == 1
    assert result["stake"] == 100
    assert result["return"] == 250
    assert result["roi"] == 2.5
