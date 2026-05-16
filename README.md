# Tensor Network Engine

Implements Matrix Product State (MPS) / Tensor Train decomposition for ETF return prediction. Builds a 3‑order tensor (time × ETF features × macro features), compresses it via TT‑SVD, and uses the flattened cores as features in a linear regression to predict next‑day returns.

- **TT rank:** 10 (compression level)
- **Window:** 60 days
- **Output:** top 3 ETFs per universe by predicted return
- **Dashboard:** shows top ETFs and full ranking table

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
