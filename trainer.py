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
        if returns.empty or len(returns) < min(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Get macro data (only columns that exist)
        available_macro = [c for c in config.MACRO_COLUMNS if c in df.columns]
        macro = df[available_macro].copy() if available_macro else pd.DataFrame()
        if macro.empty:
            print("  No macro data; using zeros")
            macro = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + config.TENSOR_WINDOW + 10:
                print(f"  Skipping window {win}d (insufficient data: need at least {win + config.TENSOR_WINDOW + 10} days)")
                continue
            print(f"  Processing window {win}d...")
            # Train models on this window (using only the last `win` days of data)
            # We'll slice returns and macro to the last `win` days
            returns_win = returns.iloc[-win:]
            macro_win = macro.iloc[-win:] if not macro.empty else pd.DataFrame(0, index=returns_win.index, columns=config.MACRO_COLUMNS)
            models = train_tensor_predictor(returns_win, macro_win, tickers,
                                            window=config.TENSOR_WINDOW,
                                            rank=config.TT_RANK)
            if not models:
                print(f"    No models trained for window {win}d")
                continue
            # Predict for each ETF using the most recent data from the window
            etf_pred = {}
            for etf in tickers:
                pred = predict_next_return(models, returns_win, macro_win, etf, config.TENSOR_WINDOW, config.TT_RANK)
                if not np.isnan(pred):
                    etf_pred[etf] = pred
            window_results[win] = etf_pred
            for etf, pred in etf_pred.items():
                if etf not in best_per_etf or pred > best_per_etf[etf][0]:
                    best_per_etf[etf] = (pred, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (mean_ret, 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        # Store full scores for all ETFs
        full_scores = {ticker: {"score": score, "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "pred_return": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/tensor_net_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Tensor Network Engine (multi‑window) complete ===")

if __name__ == "__main__":
    main()
