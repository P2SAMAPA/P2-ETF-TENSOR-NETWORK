import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from tensor_network import train_tensor_predictor, predict_next_return

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Tensor Network) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.TENSOR_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro = data_manager.get_macro_data(df)
        if macro.empty:
            print("  No macro data – using dummy")
            macro = pd.DataFrame(0, index=returns.index, columns=["VIX", "DXY", "T10Y2Y", "TBILL_3M"])

        # Train tensor predictors for each ETF
        models = train_tensor_predictor(returns, macro, tickers, window=config.TENSOR_WINDOW, rank=config.TT_RANK)
        if not models:
            print("  No models trained")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Predict next day return for each ETF
        predictions = {}
        for ticker in tickers:
            pred = predict_next_return(models, returns, macro, ticker, config.TENSOR_WINDOW)   # <-- removed rank argument
            if not np.isnan(pred):
                predictions[ticker] = pred

        if not predictions:
            print("  No predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        sorted_etfs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, pred in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "pred_return": float(pred)})
            full_scores[ticker] = float(pred)
        print(f"  Top 3 ETFs by tensor‑predicted return: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/tensor_net_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Tensor Network Engine complete ===")

if __name__ == "__main__":
    main()
